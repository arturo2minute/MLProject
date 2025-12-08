# File Goals
# - load the generated CSV files into a data frame.  
# - the Truth column contains the ground truth for each play.
# - normalizes the lengths of the loaded vectors (not all the plays are the same length)
# - constructs a LSTM classifier
# - trains the classifier
# - serializes the classifier model to a file


import pandas as pd
import numpy as np
import os, sys, json
import argparse
import json
import logging
from datetime import datetime
import shutil

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import save_model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, Callback
from sklearn.utils.class_weight import compute_class_weight

#==================================================== DEBUGGING ====================================================

stagingDir = 'C:\\Users\\arturo.diaz\\Documents\\PLAYTYPE_EXPERIMENT\\STAGING\\'
modelDir = 'C:\\Users\\arturo.diaz\\Documents\\GitHub\\MLProject\\MODEL\\'

#==================================================== GLOBALS ======================================================

MAX_SAMPLES = 150

EPOCHS = 200

BATCH_SIZE = 32

LEARNING_RATE_ADJUSTMENT_EPOCH = 50

PATIENCE = 40

TEST_SPLIT = 0.10

LSTM_UNITS = 64

DENSE_UNITS = 128

DROPOUT = 0.3

LEARNING_RATE = 1e-3

SEED = 42

#========================================== Classes and Helper Methods =============================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare training data from video files."
    )

    # Dev / Prod flags (mutually exclusive)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dev", action="store_true", help="Run in dev mode")
    mode.add_argument("--prod", action="store_true", help="Run in prod mode")

    parser.add_argument("-i", "--input", help="Path to input directory containing csv files")
    parser.add_argument("-o", "--output", help="Path to output / staging directory")

    return parser.parse_args()

def load_and_normalize_data(directory, max_samples, feature_cols=None):
    all_X = []
    all_y = []
    fileNames = []
    
    print("Loading and normalizing data from:", directory)
    for file_name in sorted(os.listdir(directory)):
        if not file_name.endswith('.csv'):
            continue

        csv_path = os.path.join(directory, file_name)
        df = pd.read_csv(csv_path)

        if df.empty:
            print(f"Warning: {csv_path} is empty, skipping.")
            continue

        # Frame index as the feature.
        # Decide feature columns once (first file)
        if feature_cols is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [c for c in numeric_cols if c != 'Truth']
            # print("Using feature columns:", feature_cols)

        # Gives matrix of all rows in CSV
        features = df[feature_cols].values

        # Pad/truncate this play to max_samples timesteps
        X = pad_sequences(
            [features],
            maxlen=max_samples,
            dtype='float32',
            padding='post',
            truncating='post'
        )  # shape: (1, max_samples)

        all_X.append(X)
        all_y.append(df['Truth'].iloc[0])  # one label per play
        fileNames.append(file_name)

    if not all_X:
        raise ValueError(f"No CSV files found in {directory}")

    # Combine all plays into a single array:
    # (num_plays, max_samples)
    X = np.vstack(all_X)
    y = np.array(all_y)

    return X, y, fileNames, feature_cols

def encode_labels(labels, outputPath):
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    
    # Save the labels and their corresponding encoders to a JSON file
    label_mapping = {int(encoded_label): label for label, encoded_label in zip(encoder.classes_, encoder.transform(encoder.classes_))}
    labels_path = os.path.join(outputPath, 'labels.json')
    with open(labels_path, 'w') as f:
        json.dump(label_mapping, f)
            
    return to_categorical(encoded_labels)

def build_lstm(input_shape, num_classes):
    model = Sequential()
    model.add(Bidirectional(
        LSTM(64, return_sequences=True),
        input_shape=input_shape
    ))
    model.add(Dropout(DROPOUT))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(DROPOUT))
    model.add(Dense(DENSE_UNITS, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(DROPOUT))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=LEARNING_RATE),
        metrics=['accuracy']
    )
    return model

def lrSchedule(epoch, lr):
    if epoch < LEARNING_RATE_ADJUSTMENT_EPOCH:
        return float(lr)
    else:
        return float(lr * tf.math.exp(-0.1).numpy())

    # Custom callback to log output
class LoggingCallback(Callback):
    def __init__(self, logger):
        self.logger = logger

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.logger.info(f'Epoch {epoch + 1}: {logs}')

class CustomEarlyStopping(EarlyStopping):
    def __init__(self, patience=0, **kwargs):
        super().__init__(patience=patience, **kwargs)
        self.patience = patience

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print(f"INFO: early stopping training because of no observed loss function improvement in {self.patience} epochs")

#
#============================================ Main Driver code ===========================================
#

# Main function
def main():
    print("------- createTrainLSTM.py -------")
    
    # Gather arguments
    args = parse_args()

    if(args.prod == True):
        staging_root = args.input
        model_root = args.output
    else:
        staging_root = stagingDir
        model_root = modelDir

    trainDir = os.path.join(staging_root, 'TRAIN')
    validDir = os.path.join(staging_root, 'VAL')

    if not os.path.isdir(trainDir) or not os.path.isdir(validDir):
        print(f"Error: TRAIN/VAL directories not found under {staging_root}")
        exit(0)

    # Set up logging
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_150t_43f_bilstm_classw"
    run_dir = os.path.join(model_root, "runs", run_id)
    os.makedirs(run_dir, exist_ok=True)

    outputPath = run_dir

    # Logging: file + stdout
    logPath = os.path.join(outputPath, "training_log.txt")
    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.FileHandler(logPath), logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger(__name__)

    # Copy splits.json into the run dir for full traceability
    splits_src = os.path.join(model_root, "splits.json")
    splits_dst = os.path.join(outputPath, "splits.json")
    if os.path.exists(splits_src):
        shutil.copy2(splits_src, splits_dst)
    else:
        logger.info(f"Warning: splits.json not found in {model_root}; run will not record split file.")
    # Normalize each sequence to MAX_SAMPLES
    X_train, y_train, trainFiles, feature_cols = load_and_normalize_data(trainDir, MAX_SAMPLES)
    X_val,   y_val,   valFiles, _ = load_and_normalize_data(validDir,  MAX_SAMPLES, feature_cols=feature_cols)
    
    # Save feature columns for inference
    with open(os.path.join(outputPath, "feature_cols.json"), "w") as f:
        json.dump(feature_cols, f, indent=2)

    inputShape = (X_train.shape[1], X_train.shape[2])
    
    # Encode train+val together so mapping is consistent
    y_all = np.concatenate([y_train, y_val])
    # Add unique labels to MODEL dir
    y_all_cat = encode_labels(y_all, outputPath) # one-hot

    # Encode classifications for all samples [1,0,0,0], [0,1,0,0]
    y_train_cat = y_all_cat[:len(y_train)]
    y_val_cat   = y_all_cat[len(y_train):]

    numClasses = y_train_cat.shape[1]
    classNames = np.unique(y_all)
    logger.info(f"Class names: {classNames}")

    # Convert one-hot to integer labels for weighting
    y_train_int = np.argmax(y_train_cat, axis=1)

    # Compute Class weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train_int),
        y=y_train_int
    )
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}
    logger.info(f"Class weights: {class_weight_dict}")

    # Doc Config
    config = {
        "run_id": run_id,
        "max_samples": MAX_SAMPLES,
        "architecture": {
            "type": "BiLSTM",
            "lstm_units": LSTM_UNITS,
            "dense_units": DENSE_UNITS,
            "dropout": DROPOUT,
        },
        "training": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "class_weight": class_weight_dict,
            "patience": PATIENCE,
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "seed": SEED,
        },
        "data": {
            "staging_root": staging_root,
            "train_dir": trainDir,
            "val_dir": validDir,
            "train_count": int(X_train.shape[0]),
            "val_count": int(X_val.shape[0]),
        },
        "labels": sorted(list(map(str, classNames)))
    }

    config_path = os.path.join(outputPath, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # Build model
    model = build_lstm(inputShape, numClasses)

    # Doc Callbacks
    logger.info(f"Model input shape: {inputShape}")
    logger.info(f"Number of training samples: {X_train.shape[0]}")
    logger.info(f"Number of validation samples: {X_val.shape[0]}")

    earlyStopping = CustomEarlyStopping(patience=PATIENCE, min_delta=0.0001)
    loggingCallback = LoggingCallback(logger)

    # Train
    history = model.fit(
        X_train,
        y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict,
        verbose=1,
        callbacks=[earlyStopping, loggingCallback]
    )

    # Save model
    modelPath = os.path.join(outputPath, 'model.keras')
    save_model(model, modelPath)

    # Doc model
    metrics_val_path = os.path.join(outputPath, "metrics_val.json")
    with open(metrics_val_path, "w") as f:
        json.dump(history.history, f, indent=2)

    # Doc train files
    trainFilesPath = os.path.join(outputPath, 'train.txt')
    with open(trainFilesPath, 'w') as f:
        f.write('\n'.join([os.path.basename(fpath) for fpath in trainFiles]))

    # Doc val files
    valFilesPath = os.path.join(outputPath, 'valid.txt')
    with open(valFilesPath, 'w') as f:
        f.write('\n'.join([os.path.basename(fpath) for fpath in valFiles]))

    logger.info(f"")
    logger.info(f"Model saved to {modelPath}")
    logger.info(f"Training files listed in: {trainFilesPath}")
    logger.info(f"Validation files listed in: {valFilesPath}")
    logger.info(f"------- Finished -------")
    logger.info(f"")

if __name__ == '__main__':
    main()
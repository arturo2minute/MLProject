# File Goals
# - load the generated CSV files into a data frame.  
# - the Truth column contains the ground truth for each play.
# - normalizes the lengths of the loaded vectors (not all the plays are the same length)
# - constructs a LSTM classifier
# - trains the classifier
# - serializes the classifier model to a file


import pandas as pd
import numpy as np
import os, sys
import argparse
import json
import logging
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

#==================================================== DEBUGGING ====================================================

stagingDir = 'C:\\Users\\arturo.diaz\\Documents\\PLAYTYPE_EXPERIMENT\\STAGING\\'
modelDir = 'C:\\Users\\arturo.diaz\\Documents\\GitHub\\MLProject\\MODEL\\'

#==================================================== GLOBALS ======================================================
#MAX_SAMPLES = 360  # Normalize each sequence to 360 samples
MAX_SAMPLES = 150

EPOCHS = 200

BATCH_SIZE = 32

LEARNING_RATE_ADJUSTMENT_EPOCH = 50

PATIENCE = 40

TEST_SPLIT = 0.10    

#========================================== Classes and Helper Methods =============================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare training data from video files."
    )

    # Dev / Prod flags (mutually exclusive)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dev", action="store_true", help="Run in dev mode")
    mode.add_argument("--prod", action="store_true", help="Run in prod mode")

    parser.add_argument("-i", "--input", help="Path to input directory containing .mp4 files")
    parser.add_argument("-o", "--output", help="Path to output / staging directory")

    return parser.parse_args()

def load_and_normalize_data(directory, max_samples):
    all_X = []
    all_y = []
    fileNames = []
    
    print("Loading and normalizing training data from:", directory)
    for file_name in sorted(os.listdir(directory)):
        if not file_name.endswith('.csv'):
            continue

        csv_path = os.path.join(directory, file_name)
        df = pd.read_csv(csv_path)

        if df.empty:
            print(f"Warning: {csv_path} is empty, skipping.")
            continue

        # Frame index as the feature.
        #TODO upgrade this later to include richer features.
        seq = df['Frame'].values

        # Pad/truncate this play to max_samples timesteps
        X = pad_sequences(
            [seq],
            maxlen=max_samples,
            dtype='int32',
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

    return X, y, fileNames

def encode_labels(labels, outputPath):
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    
    # Save the labels and their corresponding encoders to a JSON file
    label_mapping = {int(encoded_label): label for label, encoded_label in zip(encoder.classes_, encoder.transform(encoder.classes_))}
    outputPath = outputPath + 'labels.json'
    with open(outputPath, 'w') as f:
        json.dump(label_mapping, f)
            
    return to_categorical(encoded_labels)

def build_lstm(input_shape, num_classes):
    model = Sequential()
    model.add(LSTM(50, input_shape=input_shape, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
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
    print ("------- prepareTrainingData.py -------")
    
    # Gather arguments
    args = parse_args()

    if(args.prod == True):
        staging_root = args.input
        outputPath = args.output
    else:
        staging_root = stagingDir
        outputPath = modelDir
    
    trainDir = os.path.join(staging_root, 'TRAIN')
    validDir = os.path.join(staging_root, 'VAL')

    if not os.path.isdir(trainDir) or not os.path.isdir(validDir):
        print(f"Error: TRAIN/VAL directories not found under {staging_root}")
        exit(0)

    # Set up logging
    logPath = os.path.join(outputPath, 'training_log.txt')
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(logPath), logging.StreamHandler(sys.stdout)])
    logger = logging.getLogger()
    
    # Normalize each sequence to MAX_SAMPLES
    X_train, y_train, trainFiles = load_and_normalize_data(trainDir, MAX_SAMPLES)
    X_val,   y_val,   valFiles = load_and_normalize_data(validDir,  MAX_SAMPLES)
    
    # Reshape for LSTM input
    X_train = np.expand_dims(X_train, -1)
    X_val   = np.expand_dims(X_val,   -1)
    
    inputShape = (X_train.shape[1], X_train.shape[2])
    
    # Label encoding: encode train+val together so mapping is consistent
    y_all = np.concatenate([y_train, y_val])
    y_all_cat = encode_labels(y_all, outputPath)  # one-hot

    y_train_cat = y_all_cat[:len(y_train)]
    y_val_cat   = y_all_cat[len(y_train):]

    numClasses = y_train_cat.shape[1]
    classNames = np.unique(y_all)
    print("Class names:", classNames)
    print("Num classes:", numClasses)

    # Build model
    model = build_lstm(inputShape, numClasses)

    # Callbacks
    logger.info(f"Model input shape: {inputShape}")
    logger.info(f"Number of training samples: {X_train.shape[0]}")
    logger.info(f"Number of validation samples: {X_val.shape[0]}")

    earlyStopping = CustomEarlyStopping(patience=PATIENCE, min_delta=0.0001)
    loggingCallback = LoggingCallback(logger)
    # lrScheduler = LearningRateScheduler(lrSchedule)

    # Train
    history = model.fit(
        X_train,
        y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        callbacks=[earlyStopping, loggingCallback]  # add lrScheduler if you want
    )

    # Save model
    modelPath = os.path.join(outputPath, 'model.keras')
    save_model(model, modelPath)

    # Save file lists for reference
    trainFilesPath = os.path.join(outputPath, 'train.txt')
    with open(trainFilesPath, 'w') as f:
        f.write('\n'.join([os.path.basename(fpath) for fpath in trainFiles]))

    valFilesPath = os.path.join(outputPath, 'valid.txt')
    with open(valFilesPath, 'w') as f:
        f.write('\n'.join([os.path.basename(fpath) for fpath in valFiles]))

    print("")
    print("Model saved to ", modelPath)
    print("Training files listed in:   ", trainFilesPath)
    print("Validation files listed in: ", valFilesPath)
    print("")

if __name__ == '__main__':
    main()
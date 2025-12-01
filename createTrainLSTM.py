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

#NOTE: Only enable DEVELOPER_DEBUGGING_MODE if the debugger is attached and this is the main script (vs. being included by a parent script like Bobby.py)
DEVELOPER_DEBUGGING_MODE = True if sys.gettrace() is not None and __name__ == '__main__' else False

stagingDir = 'C:\\Users\\ArturoD\\Documents\\PLAYTYPE_EXPERIMENT\\STAGING\\TRAIN\\'
modelDir = 'C:\\Users\\ArturoD\\Documents\\PLAYTYPE_EXPERIMENT\\MODEL\\'

#==================================================== GLOBALS ======================================================
#MAX_SAMPLES = 360  # Normalize each sequence to 360 samples
MAX_SAMPLES = 150

EPOCHS = 200

BATCH_SIZE = 32

LEARNING_RATE_ADJUSTMENT_EPOCH = 50

PATIENCE = 40

TEST_SPLIT = 0.10    

#========================================== Classes and Helper Methods =============================================

# Load and normalize individual CSV files
def load_and_normalize_data(directory, max_samples):
    all_X = []
    all_y = []
    fileNames = []
    
    print("Loading and normalizing training data from:", directory)
    for file_name in os.listdir(directory):
        if file_name.endswith('.csv'):
            df = pd.read_csv(os.path.join(directory, file_name))
            X = pad_sequences([df['Frame'].values], maxlen=max_samples, dtype='int32', padding='post', truncating='post')
            all_X.append(X)
            all_y.extend([df['Truth'].iloc[0]] * len(X))  # Assuming all rows in a CSV file have the same Truth label

            fileNames.append(file_name)
            
    # Combine all sequences and labels
    return np.vstack(all_X), all_y, fileNames


# Encode labels
def encode_labels(labels, outputPath):
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    
    # Save the labels and their corresponding encoders to a JSON file
    label_mapping = {int(encoded_label): label for label, encoded_label in zip(encoder.classes_, encoder.transform(encoder.classes_))}
    outputPath = outputPath + 'labels.json'
    with open(outputPath, 'w') as f:
        json.dump(label_mapping, f)
            
    return to_categorical(encoded_labels)

# Build LSTM Model
def build_lstm(input_shape, num_classes):
    model = Sequential()
    model.add(LSTM(50, input_shape=input_shape, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

#
# Learning rate schedule
#
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

# Custom early stopping callback
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
    
    if(DEVELOPER_DEBUGGING_MODE == True):
        args = {}
        args['input']   = stagingDir
        args['output']  = modelDir
    else:
        ap = argparse.ArgumentParser()
        ap.add_argument("-i", "--input", required=True, help="path to staging directory")
        ap.add_argument("-o", "--output", required=True, help="path to output directory for model")
        args = vars(ap.parse_args())
        
    data_directory = args.get('input')
    outputPath = args.get('output')
    #model_path = outputPath + 'model.h5'
    
    if not os.path.isdir(data_directory):
        print(f"Error: The input directory {data_directory} does not exist.")
        exit(0)
    
    trainDir = os.path.join(stagingDir, 'train')
    validDir = os.path.join(stagingDir, 'valid')

    if(os.path.exists(trainDir) == True):
        shutil.rmtree(trainDir)
    
    if(os.path.exists(validDir) == True):
        shutil.rmtree(validDir)
    
    #
    # Set up logging
    #
    logPath = os.path.join(outputPath, 'training_log.txt')
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(logPath), logging.StreamHandler(sys.stdout)])
    logger = logging.getLogger()
    
    #
    # Normalize each sequence to MAX_SAMPLES
    #
    X, y, fileNames = load_and_normalize_data(data_directory, MAX_SAMPLES)
    
    #
    # Reshape for LSTM input
    #
    X = np.expand_dims(X, -1)
    inputShape = (X.shape[1], X.shape[2])
    
    #
    # Get the list of unique names from the array y
    #
    classNames = np.unique(y)
    print("Class names:", classNames)

    y = encode_labels(y, outputPath)
    numClasses = y.shape[1]
    
    # Split data and file names into training and validation sets
    X_train, X_val, y_train, y_val, trainFiles, valFiles = train_test_split(X, y, fileNames, test_size=TEST_SPLIT, random_state=42)
    
    
    #
    # Construct our model
    #
    model = build_lstm(inputShape, numClasses)
    model.fit(X, y, epochs=50, batch_size=32, verbose=1)
    
    
    #
    # Construct our model
    #
    #model = buildLSTM(inputShape, numClasses, modelSize=MODEL_SIZE_SMALL)
    #model = buildLSTM(inputShape, numClasses, modelSize=MODEL_SIZE_NANO)
    
    # Callbacks for early stopping and learning rate scheduling
    earlyStopping = CustomEarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
    lrScheduler = LearningRateScheduler(lrSchedule)
    loggingCallback = LoggingCallback(logger)

    # Training
    #DISABLED history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1, callbacks=[earlyStopping, loggingCallback]) #DISABLED custom scheduler lrScheduler])
    
    modelPath = os.path.join(outputPath, 'runPass_model.keras')
    save_model(model, modelPath)
    
    #
    # Output the training and validation file names to text files
    #
    trainFilesPath = os.path.join(outputPath, 'train.txt')
    with open(trainFilesPath, 'w') as f:
        f.write('\n'.join([os.path.basename(file) for file in trainFiles]))

    valFilesPath = os.path.join(outputPath, 'valid.txt')
    with open(valFilesPath, 'w') as f:
        f.write('\n'.join([os.path.basename(file) for file in valFiles]))

    #
    # Create subdirectories in stagingDir
    #        
    os.makedirs(trainDir, exist_ok=True)
    os.makedirs(validDir, exist_ok=True)

    #
    # Copy files to respective directories
    #
    for file in trainFiles:
        shutil.copy(os.path.join(data_directory, file), trainDir)

    for file in valFiles:
        shutil.copy(os.path.join(data_directory, file), validDir)

    print("")
    print("Model saved to ", modelPath)
    print("Training files saved to:   ", trainFilesPath)    
    print("Validation files saved to: ", valFilesPath)
    print("")
    
    # Save model
    # save_model(model, model_path)
    # print("Model saved to", model_path)

if __name__ == '__main__':
    main()
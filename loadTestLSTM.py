# File Goals
# - load the serialized classifier model
# - load an unseen CSV file and format it into a vector (padding if necessary)
# - perform inference
# - output the final classification prediction and confidence


import os, sys, json
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.metrics import recall_score, confusion_matrix, f1_score

#==================================================== DEBUGGING ====================================================

#NOTE: Only enable DEVELOPER_DEBUGGING_MODE if the debugger is attached and this is the main script (vs. being included by a parent script like Bobby.py)
DEVELOPER_DEBUGGING_MODE = True if sys.gettrace() is not None and __name__ == '__main__' else False

stagingDir = 'C:\\Users\\ArturoD\\Documents\\PLAYTYPE_EXPERIMENT\\STAGING\\TEST\\'
#modelDir = 'C:\\Users\\ArturoD\\Documents\\PLAYTYPE_EXPERIMENT\\MODEL\\model.h5'
modelDir = 'C:\\Users\\ArturoD\\Documents\\PLAYTYPE_EXPERIMENT\\MODEL\\runPass_model.keras'

#========================================== Classes and Helper Methods =============================================

# Load model labels
def load_label_mappings(labels_path):
    # Construct the path to the JSON file
    json_file_path = os.path.join(labels_path, 'labels.json')
    
    # Check if the JSON file exists
    if not os.path.isfile(json_file_path):
        print(f"Error: The file {json_file_path} does not exist.")
        return None
    
    # Read the JSON file and save its contents to a dictionary
    with open(json_file_path, 'r') as f:
        label_mapping = json.load(f)
    
    return label_mapping

# Load model
def load_classifier_model(model_path): 
    return load_model(model_path)

# Load and prepare CSV data
def load_and_prepare_data(csv_file, max_samples):
    data = pd.read_csv(csv_file)
    sequences = [data['Frame'].values]  # Assume single video per CSV in inference
    padded_sequences = pad_sequences(sequences, maxlen=max_samples, dtype='int32', padding='post', truncating='post')
    return np.expand_dims(padded_sequences, -1), data['Truth'].iloc[0]  # Reshape for LSTM, return ground truth

# Perform inference
def infer(model, data):
    prediction = model.predict(data)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)
    return predicted_class, confidence

#
#============================================ Main Driver code ===========================================
#

# Main function
def main():
    print ("------- loadTestLSTM.py -------")
    
    if(DEVELOPER_DEBUGGING_MODE == True):
        args = {}
        args['input']   = stagingDir
        args['model']  = modelDir
    else:
        ap = argparse.ArgumentParser()
        ap.add_argument("-i", "--input", required=True, help="path to staging directory")
        ap.add_argument("-m", "--model", required=True, help="path to model")
        args = vars(ap.parse_args())
    
    model_path = args.get('model')
    data_directory = args.get('input')
    
    if not os.path.isfile(model_path):
        print(f"Error: The model directory {model_path} does not exist.")
        exit(0)
    if not os.path.isdir(data_directory):
        print(f"Error: The input directory {data_directory} does not exist.")
        exit(0)
    
    max_samples = 360  # Must be the same as used in training

    model = load_classifier_model(model_path)
    
    label_mapping = load_label_mappings(os.path.dirname(model_path))
    
    # Accuracy
    total_files = 0
    correct_predictions = 0
    
    # Recall
    true_labels = []
    predicted_labels = []
    
    # Walk the directory
    for dirpath, dirnames, filenames in os.walk(data_directory):
        for csv_file in filenames:
            
            if(csv_file.endswith('.csv') == False):
                # Skip non-video files
                continue
            
            data, ground_truth = load_and_prepare_data((dirpath + csv_file), max_samples)
            predicted_class, confidence = infer(model, data)
            
            ground_truth_encoded = int(list(label_mapping.keys())[list(label_mapping.values()).index(ground_truth)])
            is_correct = predicted_class == ground_truth_encoded
            correct_predictions += is_correct
            total_files += 1
            
            true_labels.append(ground_truth_encoded)
            predicted_labels.append(predicted_class)
            
            print(f'Play: {csv_file}')
            print(f'Predicted Class: {label_mapping.get(str(predicted_class))}, Confidence: {confidence:.2f}')
            
    accuracy = (correct_predictions / total_files) * 100 if total_files > 0 else 0
    print(f'Overall Accuracy: {accuracy:.2f}%')
    
    if true_labels and predicted_labels:
        recall = recall_score(true_labels, predicted_labels, average='macro')
        f1 = f1_score(true_labels, predicted_labels, average='macro')
        print(f'Recall: {recall:.2f}')
        print(f'F1 Score: {f1:.2f}')
        
        cm = confusion_matrix(true_labels, predicted_labels)
        print(f'Confusion Matrix:\n{cm}')
    else:
        print("No predictions made.")

if __name__ == '__main__':
    main()
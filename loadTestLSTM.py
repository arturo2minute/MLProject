# File Goals
# - load the serialized classifier model
# - load an unseen CSV file and format it into a vector (padding if necessary)
# - perform inference
# - output the final classification prediction and confidence


import os, sys, json
import pandas as pd
import argparse


from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.metrics import recall_score, confusion_matrix, f1_score

#==================================================== DEBUGGING ====================================================

stagingDir = 'C:\\Users\\arturo.diaz\\Documents\\PLAYTYPE_EXPERIMENT\\STAGING\\'
modelDir = 'C:\\Users\\arturo.diaz\\Documents\\GitHub\\MLProject\\MODEL\\'

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

# Load model labels
def load_label_mappings(labels_dir):
    # Construct the path to the JSON file
    json_file_path = os.path.join(labels_dir, 'labels.json')
    
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

    # Select numeric feature columns except Truth
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c != 'Truth']

    features = data[feature_cols].values  # shape (T, F)

    padded_sequences = pad_sequences(
        [features],
        maxlen=max_samples,
        dtype='float32',
        padding='post',
        truncating='post'
    )  # shape: (1, max_samples, F)

    # Ground truth is play-level label
    ground_truth = data['Truth'].iloc[0]

    return padded_sequences, ground_truth

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
    
    # Gather arguments
    args = parse_args()

    if(args.prod == True):
        staging_root = args.input
        model_dir = args.output
    else:
        staging_root = stagingDir
        model_dir = modelDir
    
    testDir = os.path.join(staging_root, 'TEST')

    if not os.path.isdir(testDir) or not os.path.isdir(model_dir):
        print(f"Error: TEST/MODEL directories not found")
        exit(0)
    
    model_path = os.path.join(model_dir, 'model.keras')
    
    # Load Label mappings from json file
    label_mapping = load_label_mappings(os.path.dirname(model_dir))
    
    # Load model
    model = load_classifier_model(model_path)

    # Build reverse mappings once
    idx_to_label = {int(k): v for k, v in label_mapping.items()}
    label_to_idx = {v: k for k, v in idx_to_label.items()}

    # Accuracy
    total_files = 0
    correct_predictions = 0
    
    # Recall
    all_ground_truths = []
    all_predictions = []

    max_samples = 150  # Must be the same as used in training
    
    # Walk the directory
    for dirpath, dirnames, filenames in os.walk(testDir):
        for csv_file in filenames:
            
            if(csv_file.endswith('.csv') == False):
                # Skip non-video files
                continue
            
            # Prepare data
            csv_path = os.path.join(dirpath, csv_file)
            data, ground_truth = load_and_prepare_data(csv_path, max_samples)
            
            ground_truth_encoded = label_to_idx[ground_truth]

            # Model prediction
            probs = model.predict(data, verbose=0)[0]
            predicted_class = int(np.argmax(probs))
            confidence = float(np.max(probs))

            all_ground_truths.append(ground_truth_encoded)
            all_predictions.append(predicted_class)

            print(
                f"{csv_file}: "
                f"GT={ground_truth} ({ground_truth_encoded})  "
                f"Pred={idx_to_label[predicted_class]} ({predicted_class})  "
                f"Conf={confidence:.2f}"
            )
            
    # Show Results
    accuracy = (correct_predictions / total_files) * 100 if total_files > 0 else 0
    print(f'Overall Accuracy: {accuracy:.2f}%')
    
    if all_ground_truths and all_predictions:
        recall = recall_score(all_ground_truths, all_predictions, average='macro')
        f1 = f1_score(all_ground_truths, all_predictions, average='macro')
        print(f'Recall: {recall:.2f}')
        print(f'F1 Score: {f1:.2f}')
        
        cm = confusion_matrix(all_ground_truths, all_predictions)
        print(f'Confusion Matrix:\n{cm}')
    else:
        print("No predictions made.")

    print ("------- Finished -------") 

if __name__ == '__main__':
    main()
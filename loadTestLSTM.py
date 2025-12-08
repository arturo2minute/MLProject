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
modelDir = 'C:\\Users\\arturo.diaz\\Documents\\GitHub\\MLProject\\MODEL\\runs\\20251208_130542_150t_43f_bilstm_classw'

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

# Load feature cols
def load_feature_cols(model_dir):
    path = os.path.join(model_dir, "feature_cols.json")
    if not os.path.isfile(path):
        print(f"Warning: feature_cols.json not found in {model_dir}; "
              "falling back to numeric-cols-except-Truth.")
        return None
    with open(path, "r") as f:
        return json.load(f)

# Load splits
def load_splits(model_dir):
    path = os.path.join(model_dir, "splits.json")
    if not os.path.isfile(path):
        print(f"Warning: splits.json not found in {model_dir}; mp4 mapping will be empty.")
        return None
    with open(path, "r") as f:
        return json.load(f)

# Load and prepare CSV data
def load_and_prepare_data(csv_file, max_samples, feature_cols=None):
    data = pd.read_csv(csv_file)

    if feature_cols is None:
        # Fallback: derive feature columns from this CSV
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
    label_mapping = load_label_mappings(model_dir)
    
    # Load model
    model = load_classifier_model(model_path)

    # Feature mappings
    feature_cols = load_feature_cols(model_dir)

    # Build reverse mappings once
    idx_to_label = {int(k): v for k, v in label_mapping.items()}
    label_to_idx = {v: k for k, v in idx_to_label.items()}

    # Load splits.json for this run so we can map CSVs back to original mp4s
    splits = load_splits(model_dir)
    mp4_map = {}
    if splits and "test" in splits:
        # Build mapping: CSV basename -> original mp4 path
        mp4_map = {
            os.path.splitext(os.path.basename(p))[0]: p
            for p in splits["test"]
        }

    # Accuracy
    total_files = 0
    correct_predictions = 0
    
    # Recall
    prediction_rows = []
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
            data, ground_truth = load_and_prepare_data(csv_path, max_samples, feature_cols)
            
            ground_truth_encoded = label_to_idx[ground_truth]

            # Model prediction
            probs = model.predict(data, verbose=0)[0]
            predicted_class = int(np.argmax(probs))
            confidence = float(np.max(probs))

            all_ground_truths.append(ground_truth_encoded)
            all_predictions.append(predicted_class)

            basename = os.path.splitext(csv_file)[0]
            mp4_path = mp4_map.get(basename, "")

            correct_flag = int(predicted_class == ground_truth_encoded)

            # Build row for CSV
            row = {
                "csv_file": csv_file,
                "mp4_path": mp4_path,
                "truth_label": ground_truth,
                "truth_idx": int(ground_truth_encoded),
                "pred_label": idx_to_label[predicted_class],
                "pred_idx": predicted_class,
                "correct": correct_flag,
                "confidence": confidence,
            }
            # Add per-class probabilities
            for idx, label in idx_to_label.items():
                row[f"prob_{label}"] = float(probs[idx])

            prediction_rows.append(row)

            print(
                f"{csv_file}: "
                f"GT={ground_truth} ({ground_truth_encoded})  "
                f"Pred={idx_to_label[predicted_class]} ({predicted_class})  "
                f"Conf={confidence:.2f}"
            )
    
    if prediction_rows:
        pred_df = pd.DataFrame(prediction_rows)
        pred_path = os.path.join(model_dir, "predictions_test.csv")
        pred_df.to_csv(pred_path, index=False)
        print(f"Saved test predictions to: {pred_path}")
    else:
        print("No test CSVs found; nothing to save.")

    # Show Results
    if all_ground_truths and all_predictions:
        gt_arr = np.array(all_ground_truths)
        pred_arr = np.array(all_predictions)

        accuracy = float((gt_arr == pred_arr).mean())      # 0–1
        recall = float(recall_score(gt_arr, pred_arr, average='macro'))
        f1 = float(f1_score(gt_arr, pred_arr, average='macro'))
        cm = confusion_matrix(gt_arr, pred_arr)

        print(f'Overall Accuracy: {accuracy * 100:.2f}%')
        print(f'Recall (macro): {recall:.2f}')
        print(f'F1 Score (macro): {f1:.2f}')
        print(f'Confusion Matrix:\n{cm}')

        metrics = {
            "split": "test",
            "accuracy": accuracy,
            "recall_macro": recall,
            "f1_macro": f1,
            "confusion_matrix": {
                "labels": [idx_to_label[i] for i in sorted(idx_to_label.keys())],
                "matrix": cm.tolist(),
            },
        }
        metrics_path = os.path.join(model_dir, "metrics_test.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        
    else:
        print("No predictions made.")

    print ("------- Finished -------") 

if __name__ == '__main__':
    main()
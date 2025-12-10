MLProject – Football Play Type Classification

# --------- Overview ---------
This repository contains the sequence-modeling part of a system that classifies American football plays into four classes:
  - OFFENSE
  - PUNT
  - KICKOFF
  - FIELDGOAL

Raw video is first processed by proprietary models (not included here) to produce per-frame features. This project then:
  1. Aggregates those frame features into per-play time series (one CSV per play).
  2. Trains a BiLSTM classifier on those sequences.
  3. Evaluates the trained model on a held-out test set and saves versioned run artifacts.

# --------- Important limitations ---------
  - No training or test video data is included.
  - The nn/ and utils/ directories are intentionally excluded because they contain proprietary models and utilities owned by 2 Minute Warning.
  - You cannot run the full data preparation pipeline end-to-end without providing your own equivalents for these components and your own data.


# --------- Current model status ---------

Most recent run directory (example):

MODEL/runs/20251208_130659_150t_43f_bilstm_classw/

Test performance for that run (held-out, unseen videos):
  - Accuracy: 86.62%
  - Macro recall: 0.85
  - Macro F1: 0.79

Qualitative summary:
  - FIELDGOAL and KICKOFF are classified very accurately.
  - PUNT is generally good, but some punts are misclassified as OFFENSE.
  - OFFENSE has decent recall but lower precision; the model tends to over-predict OFFENSE on borderline PUNT vs OFFENSE plays.


# --------- Repository layout ---------

High-level structure (only the public pieces):
MODEL/
    splits.json (train/val/test definitions for .mp4 files)
    runs/
        <run_id>/
            model.keras (trained BiLSTM model)
            labels.json (class index → label)
            feature_cols.json (feature column names used at training time)
            splits.json (copy of global split used for this run)
            config.json (basic training configuration and counts)
            metrics_val.json (per-epoch train/val metrics)
            metrics_test.json (test accuracy, macro recall/F1, confusion matrix)
            predictions_test.csv (per-play predictions on TEST)
            training_log.txt (training log)
            train.txt (list of TRAIN CSV files)
            valid.txt (list of VAL CSV files)
    prepareTrainingData.py (data prep; depends on nn/ and utils/)
    createTrainLSTM.py (train the sequence classifier)
    loadTestLSTM.py (evaluate a trained run on TEST)
    .gitignore
    README.md

    The nn/ and utils/ directories are not present in this public repository.


# --------- Dependencies ---------

This project assumes a modern Python environment and common ML libraries. Example stack:
Python 3.10+
numpy
pandas
scikit-learn
tensorflow (Keras)
onnxruntime (for ONNX models in data prep)
opencv-python (for video processing in data prep)

You can install these manually using pip in a virtual environment.

# --------- Data preparation (conceptual) ---------

prepareTrainingData.py is responsible for:
  - Reading the train/val/test split from MODEL/splits.json.
  - For each play video (.mp4):
      - Running proprietary frame-level models (from nn/ and utils/).
      - Sampling every N frames (for example, every 5th frame).
      - Building a per-play time series with:
          - ruth (play type label for the play)
          - Frame index
          - Model confidences and other numeric features
  - Writing one CSV per play into a staging directory:
      - STAGING/TRAIN/
      - STAGING/VAL/
      - STAGING/TEST/

Because required models and utilities are not part of this public repo, this script is mostly here as a reference to the expected CSV format and pipeline.


# --------- Model training ---------

createTrainLSTM.py does the following:
  - Loads CSVs from STAGING/TRAIN and STAGING/VAL.
  - Uses all numeric columns except “Truth” as features.
  - Pads or truncates each play to a fixed number of timesteps (currently 150).
  - Encodes labels (FIELDGOAL, KICKOFF, OFFENSE, PUNT) and saves the mapping to labels.json.
  - Computes class weights to compensate for label imbalance.
  - Builds and trains a BiLSTM classifier:
      - Input shape: (timesteps = 150, features ≈ 40–50 numeric columns)
      - Two Bidirectional LSTM layers with dropout
      - Dense layer with ReLU, batch normalization, dropout
      - Final softmax output layer (4 classes)
      - Optimized with Adam and early stopping on validation loss
  - Creates a new run directory under MODEL/runs/<run_id>/ and saves:
      - model.keras
      - labels.json
      - feature_cols.json
      - splits.json
      - config.json
      - metrics_val.json
      - training_log.txt
      - train.txt, valid.txt


# --------- Model evaluation ---------

loadTestLSTM.py evaluates a specific trained run on the TEST split:
  - Loads from a selected run directory:
      - model.keras
      - labels.json
      - feature_cols.json
      - splits.json
  - Loads CSVs from STAGING/TEST.
  - Reconstructs input tensors using the saved feature_cols (matching training).
  - Runs inference for each play and decodes predictions with labels.json.
  - Maps each CSV back to its original .mp4 path using the copied splits.json.
  - Writes:
      - predictions_test.csv (one row per play with truth, prediction, confidence, per-class probabilities)
      - metrics_test.json (accuracy, macro recall, macro F1, and confusion matrix)

# --------- Local Workflows for background training ---------

Git Bash example (Windows)
  cd ~
  cd Documents/GitHub/MLProject

  If using virt env, activate virtual environment (Git Bash)
  source .venv/Scripts/activate

  Prepare CSV data from videos
  python prepareTrainingData.py \
    --prod \
    --input "C:\\Users\\{user}\\Documents\\PLAYTYPE_EXPERIMENT\\SMALL_DATASET" \
    --output "C:\\Users\\{user}\\Documents\\PLAYTYPE_EXPERIMENT\\STAGING\\" \
    > prep_log.txt 2>&1

  Train BiLSTM classifier and create a new run under MODEL/runs
  python createTrainLSTM.py \
    --prod \
    --input "C:\\Users\\{user}\\Documents\\PLAYTYPE_EXPERIMENT\\STAGING\\" \
    --output "C:\\Users\\{user}\\Documents\\GitHub\\MLProject\\MODEL\\"

  Evaluate a specific run on TEST and write predictions/metrics into that run directory
  python loadTestLSTM.py \
    --prod \
    --input "C:\\Users\\{user}\\Documents\\PLAYTYPE_EXPERIMENT\\STAGING\\" \
    --output "C:\\Users\\{user}\\Documents\\GitHub\\MLProject\\MODEL\\runs\\20251208_130659_150t_43f_bilstm_classw"
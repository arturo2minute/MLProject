# File Goals
# Create pathing for Train and Testing Data
# Load static playtype classifyer (classifies playtype on a given frame)
# Load all the .mp4 play files into an aray and shuffle for randomness
# Go through all the files and for evey 5 frames classify playtype with static classify
# Load this new data into csv file and save to coorisponding location

import os, sys
import argparse
import time
import cv2
import csv
import random
import json
from collections import Counter

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "utils"))

from utils.shared.NNClassifier import *
from utils.shared.NNDetector import *
import utils.shared.DDamenUtils as DD
#==================================================== DEBUGGING ====================================================

directory = 'C:\\Users\\arturo.diaz\\Documents\\PLAYTYPE_EXPERIMENT\\SMALL_DATASET'
stagingDir = 'C:\\Users\\arturo.diaz\\Documents\\PLAYTYPE_EXPERIMENT\\STAGING\\'
showVideo = False
        
#==================================================== Constants ====================================================

CLASS_NAMES = [
    "OFFENSE",
    "KICKOFF",
    "PUNT",
    "FIELDGOAL"
]

DEFAULT_CONFIDENCE_THRESHOLD    = 0.50

HEADER_LIST = ["Truth", "Frame", "Prediction", "Confidence",
                "OL",
                "CENTER",
                "QB",
                "T",
                "F",
                "LTE",
                "LUNBTACKLE",
                "RTE",
                "RUNBTACKLE",
                "LWG1",
                "LWG2",
                "RWG1",
                "RWG2",
                "WR_R1",
                "WR_R2",
                "WR_R3",
                "WR_R4",
                "WR_R5",
                "WR_L1",
                "WR_L2",
                "WR_L3",
                "WR_L4",
                "WR_L5",
                "TE_SPLIT",
                "RBL1",
                "RBL2",
                "RBL3",
                "RBC1",
                "RBC2",
                "RBC3",
                "RBR1",
                "RBR2",
                "RBR3",
                "QBU",
                "QBS",
                "KICKOFF_KICKER",
                "PLACE_KICKER",
                "PLACE_HOLDER",
                "PUNTER",
                "KICKOFF_TEAM",
                "KICK_RETURN"
               ]

VALID_OFFENSIVE_CLASS = {
    "OL": 1,
    "CENTER": 2,
    "QB": 3,
    "T": 4,
    "F": 5,
    "LTE": 6,
    "LUNBTACKLE": 7,
    "RTE": 8,
    "RUNBTACKLE": 9,
    "LWG1": 10,
    "LWG2": 11,
    "RWG1": 12,
    "RWG2": 13,
    "WR_R1": 14,
    "WR_R2": 15,
    "WR_R3": 16,
    "WR_R4": 17,
    "WR_R5": 18,
    "WR_L1": 19,
    "WR_L2": 20,
    "WR_L3": 21,
    "WR_L4": 22,
    "WR_L5": 23,
    "TE_SPLIT": 24,
    "RBL1": 25,
    "RBL2": 26,
    "RBL3": 27,
    "RBC1": 28,
    "RBC2": 29,
    "RBC3": 30,
    "RBR1": 31,
    "RBR2": 32,
    "RBR3": 33,
    "QBU": 34,
    "QBS": 35,
    "KICKOFF_KICKER": 36,
    "PLACE_KICKER": 37,
    "PLACE_HOLDER": 38,
    "PUNTER": 39,
    "KICKOFF_TEAM": 40,
    "KICK_RETURN": 41
}
    
#========================================== Classes and Helper Methods =============================================

class NNType:
	def __init__(self):
		self.DNN_WEIGHTS 	= None;
		self.DNN_CONFIG 	= None;
		self.DNN_NAMES		= None;
		self.DNN_CLASSNAMES	= None;
		self.DNN_MODEL_VERSION = None;

	def __repr__(self):
		string = "NN Model: DNN_WEIGHTS=[{0}], DNN_CONFIG={1}".format(self.DNN_WEIGHTS, self.DNN_CONFIG)

		return string;

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

def create_and_write_csv(directory, file_path, data):

    # Ensure the directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)
        
     # Splitting the file path to get the file name and extension
    file_name, _ = os.path.splitext(os.path.basename(file_path))

    # Full path to the CSV file
    csv_file_path = os.path.join(directory, file_name + '.csv')
    
    # Write data to the CSV file
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(HEADER_LIST)  # Write the headers as the first row
        for row in data:
            writer.writerow(row)
            
def find_class_name_in_path(videoPath):
    current_path = os.path.dirname(videoPath)
    while current_path != os.path.dirname(current_path):  # while not at the root directory
        for class_name in CLASS_NAMES:
            if class_name.lower() in current_path.lower():
                return class_name
        current_path = os.path.dirname(current_path)
    return None
            
def load_playTypeClass_data(playTypeClass, data, videoPath, currentFrame):
    # if playTypeClass is a list, take the top prediction
    if isinstance(playTypeClass, list):
        # Take the top prediction
        predictedClassObject = playTypeClass[0]
        
    prediction = predictedClassObject.label.upper() if predictedClassObject is not None else "UNKNOWN"
    confidence = predictedClassObject.confidence 	if predictedClassObject is not None else 0.0
    
    # Add row to data
    # NOTE: the ground truth value for this play is in the directory of the video file
    playTypeTruthValue = find_class_name_in_path(videoPath)
    if playTypeTruthValue is None:
        raise ValueError(f"Unable to find Truth label in path: {videoPath}")
    
    data.append([playTypeTruthValue, str(currentFrame), str(prediction), str((confidence * 100))])

def load_offenseClass_data(offenseClass, data):
    n = len(data) - 1
    tmpList = [0] * 41
    
    for obj in offenseClass:
        newObject = obj.label.replace(">", "").replace("<", "")
        if newObject in VALID_OFFENSIVE_CLASS:
            tmpList[VALID_OFFENSIVE_CLASS[newObject] - 1] = tmpList[VALID_OFFENSIVE_CLASS[newObject] - 1] + 1
    
    data[n] = data[n] + (tmpList)

def process_videos(video_files, output_directory, playTypeClassifier, offenseDetector, debug, videoPlayer, SAMPLE_RATE):
    
    for videoPath in video_files:

        vs = cv2.VideoCapture(videoPath)
        videoPlayer.initWithVideoSource(vs, 0)
        
        prediction = "UNKNOWN"
        confidence = 0.0
        
        data = []
        offenseClass = []

        print("Processing video {0} ...".format(videoPath))

        # Loop through current stream
        while True:
            
            try:
                # Grab the frame from the video stream
                frame = videoPlayer.getNextFrame()
                currentFrame = videoPlayer.currentFrameNumber()
                # Check to see if we have reached the end of the stream
                if frame is None:
                    break
                
            except Exception as e:
                print("exception {0} ...".format(e))
                continue
            
            # Process the frame
            (H, W) = frame.shape[:2]
            
            if(currentFrame % SAMPLE_RATE == 0):
                # Classify
                playTypeClass = playTypeClassifier.classifyObject(frame, confidenceThreshold=DEFAULT_CONFIDENCE_THRESHOLD, returnAllClassifications=True, debug=debug)
                offenseClass = offenseDetector.detectObjects(frame, confidenceThreshold=DEFAULT_CONFIDENCE_THRESHOLD, debug=debug)

                # Load Data
                load_playTypeClass_data(playTypeClass, data, videoPath, currentFrame)
                load_offenseClass_data(offenseClass, data)

            # Show the output frame
            if showVideo:
                if offenseClass is not None:
                    for obj in offenseClass:
                        print("FRAME[{0}], DETECTED OBJECT: {1}".format(currentFrame, obj))
                        DD.drawRect(frame, obj, (0, 255, 0), obj.label)
                    print("")
                
                DD.addTextBox(frame, "FRAME[{0}]".format(currentFrame), W-270, H-10, "MEDIUM_FONT")
                DD.showImage(frame, "Offense", False, True)
            
                key = cv2.waitKey(1) & 0xFF  # 25ms was empirically chosen to give a frame rate of ~27fps

                # Runtime options..
                if key == ord("q"):
                    break
                
                if key == ord(" "):
                    cv2.waitKey(0)	
    

        print("  --> Writing {0} samples to CSV file...".format(len(data)))
        create_and_write_csv(output_directory, videoPath, data)

def load_nn(type, number, NNSize, NN_Type):
    #
	# INIT: Load the Neural Networks
	#
    NNrootDir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nn")
    modelPath = os.path.join(NNrootDir, "models", type, number, "best.onnx")
    
    NN = NNType()
    
    DNNs = { NN_Type : NN }
    
    for key in DNNs.keys():
        dnn = DNNs[key]

        dnn.DNN_WEIGHTS = modelPath
        dnn.DNN_CONFIG  = os.path.join(os.path.dirname(modelPath), 'config')
        
        print("{0} using model weights file: {1}".format(key, dnn.DNN_WEIGHTS))
        print("{0} using model config  file: {1}".format(key, dnn.DNN_CONFIG))
        
        # TODO: make this dynamically identified at some point
        dnn.DEFAULT_NETWORK_SIZE = NNSize
        print("{0} using network size: {1}".format(key, dnn.DEFAULT_NETWORK_SIZE))
        
        dnn.DNN_NAMES  = os.path.join(dnn.DNN_CONFIG, 'objects.names')
        dnn.DNN_CLASSNAMES = DD.loadClassNamesFromFile(dnn.DNN_NAMES)
        dnn.DNN_MODEL_VERSION = DD.getNNRuntimeVersion(dnn.DNN_CONFIG)
        
        if(dnn.DNN_CLASSNAMES is None or len(dnn.DNN_CLASSNAMES) == 0):
            print("CRITICAL ERROR: cannot load {0}.DNN_CLASSNAMES".format(dnn))
            
    t0 = time.time()
    
    if NN_Type == "Classifier":
        return ObjectClassifier(NN.DNN_WEIGHTS, NN.DNN_CONFIG, NN.DNN_CLASSNAMES, NN.DEFAULT_NETWORK_SIZE)

    return ObjectDetector(NN.DNN_WEIGHTS, NN.DNN_CONFIG, NN.DNN_CLASSNAMES, NN.DEFAULT_NETWORK_SIZE)

def summarize_split(files, name):
    counts = Counter()
    for path in files:
        label = find_class_name_in_path(path)
        counts[label] += 1
    print(f"\n{name} split summary ({len(files)} videos):")
    for cls_name in CLASS_NAMES:
        print(f"  {cls_name}: {counts.get(cls_name, 0)}")

#
#============================================ Main Driver code ===========================================
#

#
# main()
#
def main():
    print ("------- prepareTrainingData.py -------")

    args = parse_args()

    if(args.prod == True):
        input_path = args.input
        output_path = args.output
    else:
        input_path = directory
        output_path = stagingDir
    
    test_path = output_path + 'TEST\\'
    train_path = output_path + 'TRAIN\\'
    val_path   = output_path + 'VAL\\'
    
    if not os.path.isdir(input_path):
        print(f"Error: The input directory {input_path} does not exist.")
        exit(0)
    
    playTypeClassifier = load_nn("PLAYTYPE", "12082023", 800, "Classifier")
    offenseDetector = load_nn("OFFENSE", "03152024", 1024, "Detector")
    
    #
	# Load the video, sample frames, classify, and print out the results
	#
    confidenceThreshold = DEFAULT_CONFIDENCE_THRESHOLD
    debug = False
    SAMPLE_RATE = 5
    videoPlayer = DD.VideoPlayer()
    
    # Collect all .mp4 files in a list
    video_files = []
    for dirpath, dirnames, filenames in os.walk(input_path):
        for filename in filenames:
            if filename.endswith('.mp4'):
                video_files.append(os.path.join(dirpath, filename))
                
    # Normalize order so os.walk randomness doesn’t matter
    video_files = sorted(video_files)

    splits_path = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), "MODEL"), "splits.json")
    
    if os.path.exists(splits_path):
        # Reuse existing split (deterministic, auditable)
        with open(splits_path, "r") as f:
            splits = json.load(f)
        train_files = splits["train"]
        val_files = splits["val"]
        test_files = splits["test"]
        print(f"Loaded existing split from {splits_path}")
    else:
        # Create a new split once
        rng = random.Random(42)
        rng.shuffle(video_files)

        total = len(video_files)
        test_size = int(0.20 * total)
        val_size  = int(0.10 * total)

        # 20% test, 10% val, remaining (~70%) train
        test_files = video_files[:test_size]
        val_files  = video_files[test_size:test_size + val_size]
        train_files = video_files[test_size + val_size:]

        splits = {"train": train_files, "val":   val_files, "test":  test_files}

        os.makedirs(os.path.dirname(splits_path), exist_ok=True)
        with open(splits_path, "w") as f:
            json.dump(splits, f, indent=2)
        print(f"Wrote new 70/10/20 split to {splits_path}")
    
    summarize_split(train_files, "TRAIN")
    summarize_split(val_files,   "VAL")
    summarize_split(test_files,  "TEST")

    # Process the training videos
    process_videos(train_files, train_path, playTypeClassifier, offenseDetector, debug, videoPlayer, SAMPLE_RATE)
    
    # Process the testing videos
    process_videos(test_files, test_path, playTypeClassifier, offenseDetector, debug, videoPlayer, SAMPLE_RATE)
    
    
    print(f'Total files stagged: {len(video_files)}')
    print ("------- Finished -------")            
            
if __name__== "__main__":
  main()
        
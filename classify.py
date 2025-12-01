
import argparse
import time
import onnxruntime as rt
import cv2
import torch
import numpy as np


import os, sys
import glob
import shutil


sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "utils"))

from shared.NNClassifier import *
import os
import shared.DDamenUtils as DD

#==================================================== GLOBALS ====================================================

#NOTE: Only enable DEVELOPER_DEBUGGING_MODE if the debugger is attached and this is the main script (vs. being included by a parent script like Bobby.py)
DEVELOPER_DEBUGGING_MODE = True if sys.gettrace() is not None and __name__ == '__main__' else False

#==================================================== Constants ====================================================

DEFAULT_CONFIDENCE_THRESHOLD    = 0.50

#====================================================== Enums ======================================================


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


#
# Attempts to lookup the ground truth for the specified image (using the parent dir as the class name)
#
def lookupRunPassGroundTruthForVideoClip(imageFilePath):

	if imageFilePath is not None:

		# Split the path into components and reverse it
		pathComponents = os.path.normpath(imageFilePath).split(os.sep)
		pathComponents.reverse()

		return os.path.basename(os.path.dirname(imageFilePath)).upper()

	return None

#
# Determines if the label fuzzy matches the groundTruth
#
# For example, a RUN and POST_RUN are considered a match because they are both phases a RUN sequence
#
def doesFuzzyMatch(label, groundTruth, modelType):
	matchingDict = { 'RUNPASS' : 
				  		{ 	"RUN" 			: ["RUN", "POST_RUN"],
							'POST_RUN' 		: ['RUN', 'POST_RUN'],
							"PASS" 			: ["PASS", "THROW", "POST_CATCH"],
							'THROW' 		: ['PASS', 'THROW', 'POST_CATCH'],
							"POST_CATCH"	: ["PASS", "THROW", "POST_CATCH"],
							'UNKNOWN'		: ['UNKNOWN']
							}
					}
	
	if(modelType not in matchingDict):
		return False

	if(label in matchingDict[modelType]):
		return groundTruth in matchingDict[modelType][label]
	

#
#============================================ Main Driver code ===========================================
#

#
# main()
#
def main():
	print ("------- classify.py -------")

	if(DEVELOPER_DEBUGGING_MODE == True):

		args = {}
		args['input'] 	= "/Users/jeredaasheim/Documents/2MinuteWarning/Google Drive/PLAYTYPE_EXPERIMENT/SMALL_DATASET/FIELDGOAL/CBHS-vs-YCHS-Play-147_SL_Job1137.mp4"
		
	else:

		ap = argparse.ArgumentParser()
		ap.add_argument("-i", "--input", required=True, help="path to input video")

		args = vars(ap.parse_args())

	videoPath   = args["input"]

	if(os.path.exists(videoPath) == False):
		print("ERROR: input video path does not exist: {0}".format(videoPath))
		exit(0)

	#
	# Load the ONNX model
	#

	#
	# INIT: Load the Neural Networks
	#
	NNrootDir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'nn')

	modelPath = os.path.join(NNrootDir, "models", "PLAYTYPE", "12082023", "best.onnx")

	modelType = "playtype"
	NN = NNType()

	DNNs = { "Classifier" : NN }

	for key in DNNs.keys():
		dnn = DNNs[key]

		dnn.DNN_WEIGHTS = modelPath
		dnn.DNN_CONFIG  = os.path.join(os.path.dirname(modelPath), 'config')

		print("{0} using model weights file: {1}".format(key, dnn.DNN_WEIGHTS))
		print("{0} using model config  file: {1}".format(key, dnn.DNN_CONFIG))

		# TODO: make this dynamically identified at some point
		dnn.DEFAULT_NETWORK_SIZE = 800
		print("{0} using network size: {1}".format(key, dnn.DEFAULT_NETWORK_SIZE))

		dnn.DNN_NAMES  = os.path.join(dnn.DNN_CONFIG, 'objects.names')
		dnn.DNN_CLASSNAMES = DD.loadClassNamesFromFile(dnn.DNN_NAMES)
		dnn.DNN_MODEL_VERSION = DD.getNNRuntimeVersion(dnn.DNN_CONFIG)

		if(dnn.DNN_CLASSNAMES is None or len(dnn.DNN_CLASSNAMES) == 0):
			print("CRITICAL ERROR: cannot load {0}.DNN_CLASSNAMES".format(dnn))

	t0 = time.time()

	classifier = ObjectClassifier(NN.DNN_WEIGHTS, NN.DNN_CONFIG, NN.DNN_CLASSNAMES, NN.DEFAULT_NETWORK_SIZE)


	#
	# Load the video, sample frames, classify, and print out the results
	#
	confidenceThreshold = DEFAULT_CONFIDENCE_THRESHOLD
	showVideo = True
	debug = False

	SAMPLE_RATE = 5
	videoPlayer = DD.VideoPlayer()
	vs = cv2.VideoCapture(videoPath)
	videoPlayer.initWithVideoSource(vs, 0)

	prediction = "UNKNOWN"
	confidence = 0.0

	while True:

		# Grab the frame from the video stream
		frame = videoPlayer.getNextFrame()
		currentFrame = videoPlayer.currentFrameNumber()

		# Check to see if we have reached the end of the stream
		if frame is None:
			break

		# Process the frame
		(H, W) = frame.shape[:2]
		finalFrame = frame.copy()

		if(currentFrame % SAMPLE_RATE == 0):

			classObject = classifier.classifyObject(frame, confidenceThreshold=DEFAULT_CONFIDENCE_THRESHOLD, returnAllClassifications=True, debug=debug)

			# if classObject is a list, take the first element
			if isinstance(classObject, list):
				for c in classObject:
					print("  [{0}, confidence:{1:2.2f}%]".format(c.label, c.confidence * 100))

				# Take the top prediction
				predictedClassObject = classObject[0]

			prediction = predictedClassObject.label.upper() if predictedClassObject is not None else "UNKNOWN"
			confidence = predictedClassObject.confidence 	if predictedClassObject is not None else 0.0

			print("FRAME[{0}], LATEST PREDICTION:[{1}, {2:2.2f}%]".format(currentFrame, prediction, confidence * 100))  #if debug == True else ''


		# Show the output frame
		if showVideo:
			DD.addTextBox(frame, "FRAME[{0}], [{1}]".format(currentFrame, prediction), W-270, H-10, "MEDIUM_FONT")
			DD.showImage(frame, "RunOrPass", False, True)

			key = cv2.waitKey(1) & 0xFF  # 25ms was empirically chosen to give a frame rate of ~27fps

			# Runtime options..
			if key == ord("q"):
				break

			if key == ord(" "):
				cv2.waitKey(0)	


	elapsedTime = time.time() - t0


if __name__== "__main__":
  main()

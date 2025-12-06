# MLProject
A time-series deep learning model can learn features of an offensive formation to accurately predict the type of play executed in a video of American Football.

Packages:

Create venv
Python 3.13
opencv-python
pip-25.3
onnxruntime
pandas
tensorflow
scikit-learn

Steps on how to use:

Data Prep:
prepareTrainingData.py
- Create pathing for Train and Testing Data
- Load static playtype classifyer (classifies playtype on a given frame)
- Load all the .mp4 play files into an aray and shuffle for randomness
- Go through all the files and for evey 5 frames classify playtype with static classify
- Load this new data into csv file with columns: Truth, Frame, Prediction, and Confidence and save to coorisponding location

Model Training:
createTrainLSTM.py
- load the generated CSV files into a data frame.  
- Truth column contains the ground truth for each play.
- Normalizes the lengths of the loaded vectors (not all the plays are the same length)
- Constructs a LSTM classifier
- Trains the classifier
- Serializes the classifier model to a file

Model Testing:
loadTestLSTM.py
- Load the serialized classifier model
- Load an unseen CSV file and format it into a vector (padding if necessary)
- Perform inference
- Output the final classification prediction and confidence
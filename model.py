import os.path
from generate_data import text_generator_ngram
from generate_data.generator_text import generate_sample
from ultralytics import YOLO

from pathlib import Path
TRAIN_SIZE = 5000
VAL_SIZE = 2000
FOLDER = 'train'
SCRIPT_NAME = 'script-'
DET_MODEL_NAME = "yolov8s"
IMAGE_PATH = os.path.join(FOLDER, SCRIPT_NAME)

# generate data and save them
for idx, iter in enumerate(range( TRAIN_SIZE)):
    generate_sample(FOLDER, SCRIPT_NAME + str(iter))

# split into train and val

# But it's important to activate the option resume to train from the last state of that model (lr and other features).
# An example could be:
#
# model = YOLO("last.pt")
# results = model.train(epochs=200, save_period = 10, resume=True)

# Create a new YOLO model from scratch
# model = YOLO('yolov8s.yaml')

# Load a pretrained YOLO model (recommended for training)
model = YOLO('yolov8s.pt')

# Train the model using the 'coco128.yaml' dataset for 3 epochs
results = model.train(data='coco128.yaml', epochs=3)

# Evaluate the model's performance on the validation set
results = model.val()

# Perform object detection on an image using the model
results = model('https://ultralytics.com/images/bus.jpg')

# Export the model to ONNX format
success = model.export(format='onnx')
import os.path

from ultralytics import YOLO
from generate_data.generator_text import generate_sample
from pathlib import Path
TRAIN_SIZE = 5000
VAL_SIZE = 2000
FOLDER = 'data'
SCRIPT_NAME = 'script-'
DET_MODEL_NAME = "yolov8s"
IMAGE_PATH = os.path.join(FOLDER, SCRIPT_NAME)
det_model = YOLO(f'{DET_MODEL_NAME}.pt')
label_map = det_model.model.names

res = det_model(IMAGE_PATH)

# object detection model
det_model_path = Path(f"{DET_MODEL_NAME}_openvino_model/{DET_MODEL_NAME}.xml")
if not det_model_path.exists():
    det_model.export(format="openvino", dynamic=True, half=False)

# generate data and save them
for iter in enumerate(range( TRAIN_SIZE)):
    generate_sample(FOLDER, SCRIPT_NAME)

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
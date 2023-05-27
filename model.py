import os
from generate_data import text_generator_ngram
from generate_data.generator_text import generate_sample
from ultralytics import YOLO
import numpy as np
import optuna
# from sklearn.externals import joblib
import joblib
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_contour
import matplotlib.pyplot as plt
# from ultralytics.yolo.utils.metrics import ConfusionMatrix
from pathlib import Path
TRAIN_SIZE = 5
VAL_SIZE = 2
TEST_SIZE = 2
FOLDER_TRAIN = 'train'
FOLDER_VAL = 'val'
FOLDER_TEST = 'test'
SCRIPT_NAME = 'script-'
DET_MODEL_NAME = "yolov8s"
MAX_TXT_LENGTH = 5
PATH = os.getcwd()
# IMAGE_PATH = os.path.join(FOLDER, SCRIPT_NAME)

def produce_data():

    # generate data and save them
    for idx, iter in enumerate(range( TRAIN_SIZE)):
        TEXT_LENGTH = 100 * np.random.randint(1, MAX_TXT_LENGTH, size=1)[0]
        generate_sample(FOLDER_TRAIN, SCRIPT_NAME + str(iter), text_length=TEXT_LENGTH)

    # split into train and val
    for idx, iter in enumerate(range( VAL_SIZE)):
        TEXT_LENGTH = 100 * np.random.randint(1, MAX_TXT_LENGTH, size=1)[0]
        generate_sample(FOLDER_VAL, SCRIPT_NAME + str(iter+TRAIN_SIZE+1), text_length=TEXT_LENGTH)
    
    for idx, iter in enumerate(range( TEST_SIZE)):
        TEXT_LENGTH = 100 * np.random.randint(1, MAX_TXT_LENGTH, size=1)[0]
        generate_sample(FOLDER_TEST, SCRIPT_NAME + str(iter+TRAIN_SIZE+ VAL_SIZE + 1), text_length=TEXT_LENGTH)

def set_optuna_study():
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(), direction='maximize')
    study.optimize(train_model, n_trials=20, direction='maximize')
    joblib.dump(study, os.path.join('studies','handwriting_optuna.pkl'))

def results():
    study = joblib.load(os.path.join('studies','handwriting_optuna.pkl'))
    df = study.trials_dataframe().drop(['state','datetime_start','datetime_complete','system_attrs'], axis=1)
    df.head(10)
    plot_optimization_history(study)
    plot_intermediate_values(study)
    plot_contour(study)
    plot_param_importances(study)
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

# But it's important to activate the option resume to train from the last state of that model (lr and other features).
# An example could be:
#
# model = YOLO("last.pt")
# results = model.train(epochs=200, save_period = 10, resume=True)

# Create a new YOLO model from scratch
# model = YOLO('yolov8s.yaml')


# Train the model using the 'coco128.yaml' dataset for 3 epochs
# results = model.train(data='coco128.yaml', epochs=3)

# Evaluate the model's performance on the validation set
# results = model.val()

# Perform object detection on an image using the model
# results = model('https://ultralytics.com/images/bus.jpg')

# Export the model to ONNX format
# success = model.export(format='onnx')

def train_model(trial):
    # Load a pretrained YOLO model (recommended for training)
    model = YOLO('yolov8n.pt')
  
    cfg = { 'train_batch_size' : 2,
            'test_batch_size' : 1,
            'n_epochs' : 1,
            'lr'       : trial.suggest_loguniform('lr', 1e-3, 1e-2),          
            'momentum' : trial.suggest_uniform('momentum', 0.4, 0.99),
            'optimizer': trial.suggest_categorical(['RMSProp', 'Adam']),
            'cos_lr': trial.suggest_categorical([True, False]),
            'patience': 200,
            'save_period': 10}

    #   train_loader, test_loader = get_mnist_loaders(cfg['train_batch_size'], cfg['test_batch_size'])
    # Training.
    results = model.train(
    data='config.yaml',
    imgsz=640,
    epochs=cfg['n_epochs'],
    batch=cfg['train_batch_size'],
    patience = cfg['patience'],
    save_period = cfg['save_period'],
    # device  = 0 ,
    optimizer = cfg['optimizer'], # choices=['SGD', 'Adam', 'AdamW', 'RMSProp'],
    cos_lr =  cfg['cos_lr'], # double peak learning,
    momentum = cfg['momentum'],
    name='yolov8n_handwriting'
    )

    print("TRAINING RESULTS: ", results)
    # results = model.predict(conf = conf, iou = iou, save_crop=True, max_det = MAX_TXT_LENGTH * 100, )  # evaluate model performance on the test data set
    
    metrics = model.val()  # no arguments needed, dataset and settings remembered
    # metrics.box.map    # map50-95
    # metrics.box.map50  # map50
    # metrics.box.map75  # map75
    # metrics.box.maps   # a list contains map50-95 of each category
    success = YOLO("yolov8n.pt").export(format="onnx")  # export a model to ONNX
    print("TEST RESULTS: ", metrics)
    for result in metrics:
        boxes = result.boxes  # Boxes object for bbox outputs
        print("boxes ", boxes)
        # masks = result.masks  # Masks object for segmentation masks outputs
        probs = result.probs  # Class probabilities for classification outputs  
        print("probs : ", probs)

    return results

if __name__ == '__main__':
    produce_data()
    set_optuna_study()
    results()
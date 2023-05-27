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
TRAIN_SIZE = 6000
VAL_SIZE = 3000
TEST_SIZE = 1000
FOLDER_TRAIN = 'train'
FOLDER_VAL = 'val'
FOLDER_TEST = 'test'
SCRIPT_NAME = 'script-'
DET_MODEL_NAME = "yolov8x"
MAX_TXT_LENGTH = 5
PATH = os.getcwd()
# IMAGE_PATH = os.path.join(FOLDER, SCRIPT_NAME)

def produce_data():

    # generate data and save them
    for idx, iter in enumerate(range( TRAIN_SIZE)):
        TEXT_LENGTH = np.random.randint(1, 100*MAX_TXT_LENGTH, size=1)[0]
        generate_sample(FOLDER_TRAIN, SCRIPT_NAME + str(iter), text_length=TEXT_LENGTH)

    # split into train and val
    for idx, iter in enumerate(range( VAL_SIZE)):
        TEXT_LENGTH = np.random.randint(1, 100*MAX_TXT_LENGTH, size=1)[0]
        generate_sample(FOLDER_VAL, SCRIPT_NAME + str(iter+TRAIN_SIZE+1), text_length=TEXT_LENGTH)
    
    for idx, iter in enumerate(range( TEST_SIZE)):
        TEXT_LENGTH =  np.random.randint(1, 100*MAX_TXT_LENGTH, size=1)[0]
        generate_sample(FOLDER_TEST, SCRIPT_NAME + str(iter+TRAIN_SIZE+ VAL_SIZE + 1), text_length=TEXT_LENGTH)

def set_optuna_study():
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(), direction='maximize')
    study.optimize(train_model, n_trials=3)
    joblib.dump(study, os.path.join('studies','handwriting_optuna.pkl'))

def results():
    study = joblib.load(os.path.join('studies','handwriting_optuna.pkl'))
    df = study.trials_dataframe()
    print(df.head(10))
    try:
        opt_history = plot_optimization_history(study)
        opt_history.write_image(os.path.join('studies', 'opt_history.png'))
        # interm_values = plot_intermediate_values(study)
        # opt_history.write_image(os.path.join('studies', 'opt_history.png'))
        contour = plot_contour(study)
        contour.write_image(os.path.join('studies', 'contour.png'))
        param_importance = plot_param_importances(study)
        param_importance.write_image(os.path.join('studies', 'param_importance.png'))
        print("Best trial:")
    except:
        print("Plotting the optuna tuning failed")
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
  
    cfg = { 'train_batch_size' : 32,
            'test_batch_size' : 32,
            'n_epochs' : 200,
            'lr'       : trial.suggest_loguniform('lr', 1e-3, 1e-2),          
            'momentum' : trial.suggest_uniform('momentum', 0.4, 0.99),
            'optimizer': trial.suggest_categorical('optimizer', ['RMSProp', 'Adam']),
            'cos_lr': trial.suggest_categorical('cos_lr', [True, False]),
            'patience': 50,
            'save_period': 10,
            'iou': trial.suggest_uniform('iou', 0.5, 0.9),}

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
    
    metrics = model.val(data='config.yaml', save_json=True, iou=cfg['iou'])  # no arguments needed, dataset and settings remembered
    # metrics.box.map    # map50-95
    # metrics.box.map50  # map50
    # metrics.box.map75  # map75
    # metrics.box.maps   # a list contains map50-95 of each category
    # success = YOLO("yolov8n.pt").export(format="onnx")  # export a model to ONNX
    print("TEST RESULTS: ", metrics)
    print(metrics.box.map)    # map50-95
    print("DICT: ", metrics.results_dict)
    print("mean_results() ", metrics.results_dict)
    print(type(metrics.results_dict))
    fitness = dict(metrics.results_dict)['fitness']
    # metrics.box.map50  # map50
    # metrics.box.map75  # map75
    # metrics.box.maps   # a list contains map50-95 of each category

    return fitness

if __name__ == '__main__':
    produce_data()
    set_optuna_study()
    results()
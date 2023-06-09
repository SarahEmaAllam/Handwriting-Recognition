import optuna
import joblib
import os
import numpy as np
from multiprocessing import Pool
from ultralytics import YOLO
from ultralytics.yolo.utils import set_settings
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_contour
from tqdm import tqdm
from generate_data.sample_generator import generate_sample
from generate_data.data_generator import init_font
import util.helper_functions as hf
from preprocess.preprocessing import preprocessing
from util.global_params import *
from util.helper_functions import assert_dir


"""
Remove the train, test, val folders - for both images and labels
Check if the folders exist and if not create them
"""
def prep_folder_structure():
    # clean up the folders before saving new data
    dirs = ['images', 'labels']
    subdirs = ['train', 'test', 'val']

    for dir in dirs:
        for subdir in subdirs:
            hf.remove_directory(os.path.join(DATA_FOLDER, dir, subdir))

    hf.remove_directory(os.path.join(RUN_FOLDER, 'detect', STORE_NAME))

    hf.assert_dir(DATA_FOLDER)
    hf.assert_dir(DATA_FOLDER + "/images")
    hf.assert_dir(DATA_FOLDER + "/images/train")
    hf.assert_dir(DATA_FOLDER + "/images/val")
    hf.assert_dir(DATA_FOLDER + "/images/test")

    hf.assert_dir(DATA_FOLDER + "/labels")
    hf.assert_dir(DATA_FOLDER + "/labels/train")
    hf.assert_dir(DATA_FOLDER + "/labels/val")
    hf.assert_dir(DATA_FOLDER + "/labels/test")
    if os.path.exists(DATA_FOLDER + "/labels/train.cache"):
        os.remove(DATA_FOLDER + "/labels/train.cache")

    if os.path.exists(DATA_FOLDER + "/labels/val.cache"):
        os.remove(DATA_FOLDER + "/labels/val.cache")


def generate_data(idx):
    text_len = np.random.randint(1, 100 * MAX_TXT_LENGTH, size=1)[0]
    generate_sample(FOLDER_TRAIN, SCRIPT_NAME + str(idx),
                    text_length=text_len)


def split_train_val(idx):
    text_len = np.random.randint(1, 100 * MAX_TXT_LENGTH, size=1)[0]
    generate_sample(FOLDER_VAL, SCRIPT_NAME + str(idx + TRAIN_SIZE + 1),
                    text_length=text_len)


def generate_test(idx):
    text_len = np.random.randint(1, 100 * MAX_TXT_LENGTH, size=1)[0]
    generate_sample(FOLDER_TEST,
                    SCRIPT_NAME + str(idx + TRAIN_SIZE + VAL_SIZE + 1),
                    text_length=text_len)


def task_manager(func, size, name):
    # create task pool
    with Pool() as p:
        tuple(tqdm(p.imap(func, range(size)), total=size, desc=name))

    p.close()
    p.join()

    # report that all tasks are completed
    print('Done with: ' + name, flush=True)


def produce_data():

    # make sure all needed folders exist
    prep_folder_structure()

    # do preprocessing if needed
    if not os.path.exists(PREPROCESS_DIR):
        preprocessing()

    # generate data and save them
    task_manager(generate_data, TRAIN_SIZE, "Generating training data")

    # split into train and val
    task_manager(split_train_val, VAL_SIZE, "Generating validation data")

    # generate test data
    task_manager(generate_test, TEST_SIZE, "Generating testing data")


def resume_check():
    run_file = os.path.join(RUN_FOLDER, 'detect', STORE_NAME, 'weights', 'last.pt')
    return os.path.exists(run_file)


def set_optuna_study():
    storage_file = STUDIES_FOLDER + '/hwr.db'

    if not resume_check() and os.path.exists(storage_file):
        os.remove(storage_file)

    hf.assert_dir(STUDIES_FOLDER)

    study = optuna.create_study(sampler=optuna.samplers.TPESampler(),
                                direction='maximize', storage='sqlite:///' + storage_file, load_if_exists=True)
    study.optimize(train_model, n_trials=3)
    joblib.dump(study, os.path.join('studies', 'handwriting_optuna.pkl'))


def results():
    assert_dir('studies')
    study = joblib.load(os.path.join('studies', 'handwriting_optuna.pkl'))
    df = study.trials_dataframe()
    print(df.head(10))
    try:
        opt_history = plot_optimization_history(study)
        opt_history.write_image(os.path.join('studies', 'opt_history.png'))
        contour = plot_contour(study)
        contour.write_image(os.path.join('studies', 'contour.png'))
        param_importance = plot_param_importances(study)
        param_importance.write_image(
            os.path.join('studies', 'param_importance.png'))
        print("Best trial:")
    except:
        print("Plotting the optuna tuning failed")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


def train_model(trial):

    # fix to force yolo to use the directories we want it to use
    set_settings({'datasets_dir': os.path.join(os.getcwd(), DATA_FOLDER),
                  'runs_dir': os.path.join(os.getcwd(), RUN_FOLDER)})

    model_folder = MODEL_FOLDER
    det_model_name = DET_MODEL_NAME
    should_resume = resume_check()
    if should_resume:
        model_folder = os.path.join(RUN_FOLDER, 'detect', STORE_NAME, 'weights')
        det_model_name = 'last.pt'

    # Load a pretrained YOLO model (recommended for training)
    # put it in the MODEL_FOLDER folder
    path = os.getcwd()
    os.chdir(model_folder)
    model = YOLO(det_model_name)
    os.chdir(path)

    cfg = {'train_batch_size': 8,
           'test_batch_size': 8,
           'n_epochs': 200,
           'lr': trial.suggest_loguniform('lr', 1e-3, 1e-2),
           'momentum': trial.suggest_uniform('momentum', 0.4, 0.99),
           'optimizer': trial.suggest_categorical('optimizer',
                                                  ['RMSProp', 'Adam']),
           'cos_lr': trial.suggest_categorical('cos_lr', [True, False]),
           'patience': 50,
           'save_period': 10,
           'iou': trial.suggest_uniform('iou', 0.5, 0.9), }

    # Training
    results = model.train(
        data=YOLO_CONFIG,
        imgsz=640,
        epochs=cfg['n_epochs'],
        batch=cfg['train_batch_size'],
        patience=cfg['patience'],
        save_period=cfg['save_period'],
        # device = 0 ,
        optimizer=cfg['optimizer'],
        # choices = ['SGD', 'Adam', 'AdamW', 'RMSProp'],
        cos_lr=cfg['cos_lr'],  # double peak learning,
        momentum=cfg['momentum'],
        name=STORE_NAME,
        resume=should_resume
    )

    print("TRAINING RESULTS: ", results)

    metrics = model.val(data=YOLO_CONFIG, save_json=True, iou=cfg[
        'iou'])  # no arguments needed, dataset and settings remembered
    print("TEST RESULTS: ", metrics)
    print(metrics.box.map)  # map50-95
    print("DICT: ", metrics.results_dict)
    print("mean_results() ", metrics.results_dict)
    print(type(metrics.results_dict))
    fitness = dict(metrics.results_dict)['fitness']

    return fitness


def run_model(mode):

    init_font()

    # uncomment to test the data generation function (one sample)
    # generate_sample(FOLDER, "test")

    if mode == "generate":
        produce_data()
        return

    if mode == "train":
        # train the model
        set_optuna_study()

        # show the results
        results()
        return

    # generate and split the data
    if not resume_check():
        produce_data()

    # train the model
    set_optuna_study()

    # show the results
    results()




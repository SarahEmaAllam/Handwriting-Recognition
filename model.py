import optuna
import joblib
from ultralytics import YOLO
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_contour
from tqdm import tqdm
from generate_data.helper_functions import remove_train_test_val
from generate_data.sample_generator import generate_sample
from global_params import *


def produce_data():
    # clean up the folders before saving new data
    remove_train_test_val(DATA_FOLDER)

    # generate data and save them
    for idx, iter in tqdm(enumerate(range(TRAIN_SIZE)), desc="Generating training data"):
        text_len = np.random.randint(1, 100 * MAX_TXT_LENGTH, size=1)[0]
        generate_sample(FOLDER_TRAIN, SCRIPT_NAME + str(iter),
                        text_length=text_len)

    # split into train and val
    for idx, iter in tqdm(enumerate(range(VAL_SIZE)), desc="Generating validation data"):
        text_len = np.random.randint(1, 100 * MAX_TXT_LENGTH, size=1)[0]
        generate_sample(FOLDER_VAL, SCRIPT_NAME + str(iter + TRAIN_SIZE + 1),
                        text_length=text_len)

    for idx, iter in tqdm(enumerate(range(TEST_SIZE)), desc="Generating testing data"):
        text_len = np.random.randint(1, 100 * MAX_TXT_LENGTH, size=1)[0]
        generate_sample(FOLDER_TEST,
                        SCRIPT_NAME + str(iter + TRAIN_SIZE + VAL_SIZE + 1),
                        text_length=text_len)


def set_optuna_study():
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(),
                                direction='maximize')
    study.optimize(train_model, n_trials=3)
    joblib.dump(study, os.path.join('studies', 'handwriting_optuna.pkl'))


def results():
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
    # Load a pretrained YOLO model (recommended for training)
    model = YOLO(DET_MODEL_NAME)

    cfg = {'train_batch_size': 32,
           'test_batch_size': 32,
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
        data='config.yaml',
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
        name='yolov8n_handwriting'
    )

    print("TRAINING RESULTS: ", results)

    metrics = model.val(data='../config.yaml', save_json=True, iou=cfg[
        'iou'])  # no arguments needed, dataset and settings remembered
    print("TEST RESULTS: ", metrics)
    print(metrics.box.map)  # map50-95
    print("DICT: ", metrics.results_dict)
    print("mean_results() ", metrics.results_dict)
    print(type(metrics.results_dict))
    fitness = dict(metrics.results_dict)['fitness']

    return fitness


if __name__ == '__main__':
    # uncomment to test the data generation function (one sample)
    # generate_sample(FOLDER, "test")

    # generate and split the data
    produce_data()

    # train the model
    set_optuna_study()

    # show the results
    results()

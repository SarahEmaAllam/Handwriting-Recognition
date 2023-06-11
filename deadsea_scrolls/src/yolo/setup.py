from multiprocessing import Pool
from tqdm import tqdm
from generate_data.sample_generator import generate_sample
import util.helper_functions as hf
from preprocess.preprocessing import preprocessing
from util.global_params import *


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
    hf.remove_directory("studies")

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


# generate all the data
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

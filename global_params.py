# Global variables for the project
import os
import numpy as np

# Model parameters
TRAIN_SIZE = 6000
VAL_SIZE = 3000
TEST_SIZE = 1000
DET_MODEL_NAME = "yolov8m"
MAX_TXT_LENGTH = 5

# Global variables for the model
FOLDER_TRAIN = 'train'
FOLDER_VAL = 'val'
FOLDER_TEST = 'test'
SCRIPT_NAME = 'script-'


# Global variables for data generation
FOLDER = 'new'
WORD_LENGTH = 10
TEXT_LENGTH = 100 * np.random.randint(1, 5, size=1)[0]
NGRAM_SIZE = 4
Box = [float, float, float, float]
WIDTH = 640
HEIGHT = 640
PADDING = 10 * np.random.randint(10, size=1)[0]
WHITESPACE = 15

# paths (from the root directory of the project)
PATH = os.getcwd()
DATA_FOLDER = 'data'

# paths for preprocessing
SOURCE_SYMBOLS = os.path.join(DATA_FOLDER, 'monkbrill')
SOURCE_SCROLLS = os.path.join(DATA_FOLDER, 'image-data')
OUTPUT = os.path.join(DATA_FOLDER, 'preprocessed_images')

LETTERS_FOLDER = os.path.join(DATA_FOLDER, 'preprocessed_images', 'symbols')
FONT_PATH = 'Habbakuk.TTF'  # Linux is capital sensitive
N_GRAM_PATH = os.path.join('generate_data', 'ngram')

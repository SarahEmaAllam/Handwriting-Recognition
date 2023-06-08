import os
import tensorflow as tf


AUTOTUNE = tf.data.AUTOTUNE

# Relative path to the dataset
IAM_LINES_PATH = os.path.join('data/IAM-data', 'iam_lines_gt.txt')

# Relative path to the images
IMAGES_PATH = os.path.join('data', 'binarized', 'img')

# Relative path to save the preprocessed data
PREPROCESSED_DATA_PATH = os.path.join('data', 'preprocessed_data')

# parameters
MAX_LEN = 128
PADDING_TOKEN = 99

IMAGE_SIZE = (64, 512)

VAL_SPLIT = 0.2
TEST_SPLIT = 0.2

BATCH_SIZE = 32
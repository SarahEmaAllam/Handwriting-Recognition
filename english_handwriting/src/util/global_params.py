import os
import tensorflow as tf


AUTOTUNE = tf.data.AUTOTUNE

IAM_LINES_PATH = os.path.join('data', 'IAM-data', 'iam_lines_gt.txt')
IMAGES_PATH = os.path.join('data', 'IAM-data', 'img')
BINARIZED_IMAGES_PATH = os.path.join('data', 'binarized', 'img')
PREPROCESSED_DATA_PATH = os.path.join('data', 'preprocessed_data')

# parameters
MAX_LEN = 128
PADDING_TOKEN = 99

IMAGE_SIZE = (32, 256)

VAL_SPLIT = 0.2
TEST_SPLIT = 0.2

BATCH_SIZE = 64

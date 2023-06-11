import os
from preprocessing.binarization import binarize
from preprocessing.preprocessing import preprocess_test_data

from util.utils import set_working_dir
from training import train
import sys
import argparse
import tensorflow as tf
from model import Model


def load_model():
    set_working_dir(os.path.abspath(__file__))

    trained_model_path = 'logs/trained_models/model_51--207.72'
    # check if the path exists
    if not os.path.exists(trained_model_path):
        os.chdir('../')
        # print the current working directory
        print(os.getcwd())
        # load the trained model
    model = tf.keras.models.load_model(trained_model_path,
                                       custom_objects={'Model': Model})
    return model
    # print the model summary
    # model.summary()


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p", "--path", action="store_true", help="Path to line image")
    args = parser.parse_args()


if __name__ == '__main__':
    set_working_dir(os.path.abspath(__file__))
    parse()
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = input("Path to line image:")
    model = load_model()
    output, decoder = preprocess_test_data(path)
    # change working dir to root of project
    # os.chdir(os.path.dirname (os.path.abspath(__file__)))
    # train()

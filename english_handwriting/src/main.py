import os
from preprocessing.binarization import binarize
from preprocessing.preprocessing import preprocess_test_images

from util.utils import set_working_dir
from training import train
import sys
import argparse
import tensorflow as tf
from model import Model
from tensorflow import keras
from testing import decode_batch_predictions


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


def text_to_file(text, name):
    with open(name + '.txt', 'w') as f:
        for line in text:
            f.write(f"{line}\n")


if __name__ == '__main__':
    set_working_dir(os.path.abspath(__file__))
    parse()
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = input("Path to line image:")
    model = load_model()
    proc_img, decoder = preprocess_test_images(path)

    prediction_model = keras.models.Model(
        model.get_layer(name="image").input,
        model.get_layer(name="dense").output
    )

    pred_batch = prediction_model.predict(proc_img)
    predicted_text = []

    for img in pred_batch:
        predicted_line = decode_batch_predictions(pred_batch, decoder)
        predicted_text.append(predicted_line)
    text_to_file(predicted_text, path)
    # change working dir to root of project
    # os.chdir(os.path.dirname (os.path.abspath(__file__)))
    # train()

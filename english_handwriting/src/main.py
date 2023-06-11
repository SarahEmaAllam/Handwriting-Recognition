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


# loads pre-trained and saved model from disk
def load_model(model_path):
    set_working_dir(os.path.abspath(__file__))
    model = tf.keras.models.load_model(model_path,
                                       custom_objects={'Model': Model})
    return model


# reads input from console; optional
def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p", "--path", action="store_true", help="Path to line image")
    args = parser.parse_args()


# generates a txt file for each predicted line
# saves file in results diretory
# creates directoray if non-existent
def text_to_file(text, name):
    save_path = './results'
    if os.path.exists(save_path) is False:
        os.mkdir(save_path)
    for idx, line in enumerate(text):
        complete_path = os.path.join(save_path, name + str(idx + 1) + '.txt')
        with open(complete_path, 'w') as f:
            f.write(f"{line}\n")


if __name__ == '__main__':
    MODEL_PATH = ''
    set_working_dir(os.path.abspath(__file__))
    parse()
    if len(sys.argv) > 1:
        try:
            path = sys.argv[1]
        except IndexError:
            exit("Path couldn't be resolved.")
    else:
        path = input("Path to line image:")

    model = load_model(model_path=MODEL_PATH)
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

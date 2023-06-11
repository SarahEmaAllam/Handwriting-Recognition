import os
from preprocessing.preprocessing import preprocess_test_images
import sys
import argparse
import tensorflow as tf
from model import Model
from tensorflow import keras
from testing import decode_batch_predictions


# loads pre-trained and saved model from disk
def load_model(model_path):
    model = tf.keras.models.load_model(model_path,
                                       custom_objects={'Model': Model})
    return model


# reads input from console; optional
def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p", "--path", action="store_true", help="Path to line image")
    # args = parser.parse_args()

# extract filenames from given directory
# returns list of filenames
def get_filenames(path):
    all_files = os.listdir(path)
    filenames = []
    for file in all_files:
        f_name = os.path.basename(file)
        filenames.append(os.path.splitext(f_name)[0])
    return filenames


# generates a txt file for each predicted line
# saves file in 'results' directory
# creates 'results' directory in parental directory if non-existent
def text_to_file(text, path):
    save_path = './results'
    filenames = get_filenames(path)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for idx, line in enumerate(text):
        name = filenames[idx]
        complete_path = os.path.join(save_path, name + '.txt')
        with open(complete_path, 'w') as f:
            f.write(f"{line}\n")


def main():
    MODEL_PATH = 'model_46--191.00'
    # reads input either from console or as a python script
    # parse()
    if len(sys.argv) > 1:
        try:
            path = sys.argv[1]
        except IndexError:
            exit("Path couldn't be resolved.")
    else:
        path = input("Path to line image:")
        if not os.path.exists(path):
            print("Path couldn't be resolved.")
            return

    # loading pre-trained model from dsk
    model = load_model(model_path=MODEL_PATH)
    prediction_model = keras.models.Model(
        model.get_layer(name="image").input,
        model.get_layer(name="dense").output
    )

    # pre-processing images for prediciton
    proc_img, decoder = preprocess_test_images(path)
    pred_batch = prediction_model.predict(proc_img)

    # decoding and storing predicted lines as txt file
    predicted_text = decode_batch_predictions(pred_batch, decoder)
    text_to_file(predicted_text, path)


if __name__ == '__main__':
    main()

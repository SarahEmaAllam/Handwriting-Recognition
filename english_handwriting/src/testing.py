import random
from typing import List, Any

import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
import Levenshtein

from keras.src.layers import StringLookup
from model import Model
from preprocessing.preprocessing import preprocess
from util.utils import set_working_dir
from tensorflow import keras
from util.global_params import MAX_LEN


def cer(ref, hyp):
    """
    Computes the Character Error Rate (CER) between two strings.

    Arguments:
    ref -- the reference string
    hyp -- the hypothesis string

    Returns:
    The character error rate as a float.
    """

    # Create a matrix of zeros
    d = [[0 for j in range(len(hyp) + 1)] for i in range(len(ref) + 1)]

    # Initialize the first row and column of the matrix
    for i in range(len(ref) + 1):
        d[i][0] = i
    for j in range(len(hyp) + 1):
        d[0][j] = j

    # Fill in the rest of the matrix
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            if ref[i - 1] == hyp[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitution = d[i - 1][j - 1] + 1
                insertion = d[i][j - 1] + 1
                deletion = d[i - 1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    # Return the CER
    return d[len(ref)][len(hyp)] / float(len(ref))


def decode_batch_predictions(pred, decoder):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search.
    results = keras.backend.ctc_decode(
        pred, input_length=input_len, greedy=False)[0][0][:, :MAX_LEN]
    # Iterate over the results and get back the text.
    output_text = []
    for res in results:
        res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
        res = tf.strings.reduce_join(decoder(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text

def evaluate():
    # change working dir to root of project (english_handwriting)
    # os.chdir('../')
    # set_working_dir(os.path.abspath(__file__))

    trained_model_path = 'logs/trained_models/model_59--226.47'

    # check if the path exists
    if not os.path.exists(trained_model_path):
        os.chdir('../')
        # print the current working directory
        print(os.getcwd())

    # load the trained model
    model = tf.keras.models.load_model(trained_model_path,
                                       custom_objects={'Model': Model})

    # print the model summary
    model.summary()

    prediction_model = keras.models.Model(
        model.get_layer(name="image").input,
        model.get_layer(name="dense").output
    )

    _, _, test_batches, decoder = preprocess(True)

    # predictions_batches = prediction_model.predict(test_batches['image'])

    images_list = []
    predicted_labels_list = []
    true_labels_list = []

    # predict the output on one of the test batches
    for test_batch in test_batches:
        images, true_labels = test_batch['image'], test_batch['label']
        pred_batch = prediction_model.predict(images)
        predictions_text = decode_batch_predictions(pred_batch, decoder)

        for image, true_label, predicted_label in zip(images, true_labels, predictions_text):
            image = tf.keras.preprocessing.image.array_to_img(image)

            # Display the image using Matplotlib
            # plt.imshow(image, cmap='gray')
            # plt.axis('off')

            # decode the true label
            indices = tf.gather(true_label, tf.where(tf.math.not_equal(true_label, 99)))
            # Convert to string.
            true_label = tf.strings.reduce_join(decoder(indices))
            true_label = true_label.numpy().decode("utf-8")

            images_list.append(image)
            predicted_labels_list.append(predicted_label)
            true_labels_list.append(true_label)

            if random.random() < 0.1:
                print("pred:", predicted_label)
                print("actual:", true_label)
                print()

    cer_values = []
    for true_y, pred_y in zip(true_labels_list, predicted_labels_list):
        cer_values.append(cer(true_y, pred_y))

    print(np.array(cer_values).mean())

    # print("CER:", compute_cer(true_labels_list, predicted_labels_list))
    # print("WER:", compute_wer(true_labels_list, predicted_labels_list))


def testing(test_data: list[tf.Tensor], decoder: StringLookup) -> list[str]:
    """
    Test the model on the test data.
    :param test_data: the test data
    :return: the predicted labels
    """
    trained_model_path = 'logs/trained_models/model_51--207.72'

    # load the trained model
    # Load the model
    model = keras.models.load_model(trained_model_path, compile=False)

    prediction_model = keras.models.Model(
        model.get_layer(name="image").input,
        model.get_layer(name="dense").output
    )

    pred_texts = []

    for data in test_data:
        # Predict the output using the new model
        predictions = prediction_model.predict(data)
        label = decode_batch_predictions(predictions, decoder)[0]
        pred_texts.append(label)

    return pred_texts


if __name__ == '__main__':
    evaluate()

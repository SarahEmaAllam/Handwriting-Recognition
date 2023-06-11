from typing import List, Any

import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.layers import StringLookup

from model import Model
from preprocessing.preprocessing import preprocess
from util.utils import set_working_dir
from tensorflow import keras
from util.global_params import MAX_LEN


def decode_batch_predictions(pred, decoder):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search.
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :MAX_LEN
    ]
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

    # print the model summary
    model.summary()

    prediction_model = keras.models.Model(
        model.get_layer(name="image").input,
        model.get_layer(name="dense").output
    )

    train_batches, val_batches, test_batches, decoder = preprocess(
            True)

    predictions_batches = prediction_model.predict(test_batches)

    # predict the output on one of the test batches
    for test_batch, pred_batch in zip(test_batches, predictions_batches):
        images, true_labels = test_batch['image'], test_batch['label']
        predictions_text = decode_batch_predictions(pred_batch, decoder)

        for image, true_label, predicted_label in zip(images, true_labels, predictions_text):
            # print(decoder_vocab)
            # image = images[i]
            image = tf.keras.preprocessing.image.array_to_img(image)

            # true_label = true_labels[i]
            # predicted_label = str(predictions_text[i])

            # Display the image using Matplotlib
            plt.imshow(image, cmap='gray')
            plt.axis('off')

            # decode the true label
            indices = tf.gather(true_label, tf.where(tf.math.not_equal(true_label, 99)))
            # Convert to string.
            true_label = tf.strings.reduce_join(decoder(indices))
            true_label = true_label.numpy().decode("utf-8")

            print("pred:", predicted_label)
            print("actual:", true_label)

            plt.title('pred: ' + predicted_label + '\nactual: ' + true_label)
            plt.show()


def test(test_data: list[tf.Tensor], decoder: StringLookup) -> list[list[Any]]:
    """
    Test the model on the test data.
    :param test_data: the test data
    :return: the predicted labels
    """
    trained_model_path = 'logs/trained_models/model_51--207.72'

    # load the trained model
    # Load the model
    model = keras.models.load_model(trained_model_path, compile=False)

    print(model.summary())

    # Remove the label tensor from the model architecture
    # model.layers[0][-1].outbound_nodes = []

    prediction_model = keras.models.Model(
        model.get_layer(name="image").input,
        model.get_layer(name="dense").output
    )

    pred_texts = []

    for data in test_data:
        # Predict the output using the new model
        predictions = prediction_model.predict(data)

        pred_texts.append(decode_batch_predictions(predictions, decoder))

    return pred_texts


if __name__ == '__main__':
    evaluate()

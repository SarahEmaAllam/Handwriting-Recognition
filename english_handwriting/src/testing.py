import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np

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


if __name__ == '__main__':

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

    train_batches, val_batches, test_batches, decoder = preprocess(
            True)

    # predict the output on one of the test batches
    for batch in test_batches.take(1):
        predictions = model.predict(batch)
        images, true_labels = batch['image'], batch['label']
        predictions_text = decode_batch_predictions(predictions, decoder)

        for i in range(4):
            # print(decoder_vocab)
            image = images[i]
            image = tf.keras.preprocessing.image.array_to_img(image)

            true_label = true_labels[i]
            predicted_label = str(predictions_text[i])

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



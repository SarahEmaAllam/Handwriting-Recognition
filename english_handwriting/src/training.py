import datetime

import tensorflow as tf
from tensorflow import keras
import numpy as np
from util.global_params import MAX_LEN
from preprocessing import preprocessing
from model import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from data_agumentation import augment_img


def calculate_edit_distance(labels, predictions):
    # Get a single batch and convert its labels to sparse tensors.
    saprse_labels = tf.cast(tf.sparse.from_dense(labels), dtype=tf.int64)

    # Make predictions and convert them to sparse tensors.
    input_len = np.ones(predictions.shape[0]) * predictions.shape[1]
    predictions_decoded = keras.backend.ctc_decode(
        predictions, input_length=input_len, greedy=True
    )[0][0][:, :MAX_LEN]
    sparse_predictions = tf.cast(
        tf.sparse.from_dense(predictions_decoded), dtype=tf.int64
    )

    # Compute individual edit distances and average them out.
    edit_distances = tf.edit_distance(
        sparse_predictions, saprse_labels, normalize=False
    )
    return tf.reduce_mean(edit_distances)


class EditDistanceCallback(keras.callbacks.Callback):
    def __init__(self, pred_model, validation_images, validation_labels):
        super().__init__()
        self.prediction_model = pred_model
        self.validation_images = validation_images
        self.validation_labels = validation_labels

    def on_epoch_end(self, epoch, logs=None):
        edit_distances = []

        for i in range(len(self.validation_images)):
            labels = self.validation_labels[i]
            predictions = self.prediction_model.predict(
                self.validation_images[i])
            edit_distances.append(
                calculate_edit_distance(labels, predictions).numpy())

        print(
            f"Mean edit distance for epoch {epoch + 1}: {np.mean(edit_distances):.4f}"
        )


def train():
    print("[INFO] initializing model...")
    print("[INFO] compiling model...")

    train_batches, val_batches, test_batches, decoder = preprocessing.preprocess(True)
    print("[INFO] training model...")

    output_shape = decoder.vocabulary_size() + 2

    model = Model().build_model(output_shape)
    prediction_model = keras.models.Model(
        model.get_layer(name="image").input, model.get_layer(name="dense").output
    )

    # Validation data for the EditDistanceCallback.
    validation_images = []
    validation_labels = []

    for batch in val_batches:
        validation_images.append(batch['image'])
        validation_labels.append(batch['label'])

    edit_distance_callback = EditDistanceCallback(
        prediction_model, validation_images, validation_labels)

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                          histogram_freq=1)

    # train the model
    history = model.fit(
        train_batches,
        validation_data=val_batches,
        epochs=10,
        callbacks=[edit_distance_callback, tensorboard_callback]
    )

    print(history)

    # save the model to disk
    print("[INFO] saving model...")
    model.save("model.h5", save_format="h5")

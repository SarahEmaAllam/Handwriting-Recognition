import os
import time
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
# from keras.src.layers import StringLookup
from tensorflow import keras
from util.global_params import IMAGE_SIZE, MAX_LEN


class CTCLayer(keras.layers.Layer):
    def __int__(self, name=None):
        super().__init__(name=name)
        # self.loss_fn =  keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        print("y_true ", y_true)
        print("y_pred : ", y_pred)
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1),
                                              dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1),
                                              dtype="int64")
        print("input_length : ", input_length)
        print("label_length : ", label_length)
        loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_length,
                                            label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions.
        return y_pred


class Model:
    # Desired image dimensions
    img_width = IMAGE_SIZE[1]
    img_height = IMAGE_SIZE[0]

    def __int__(self):
        pass

    def build_model(self, output_shape):
        input_layer = keras.Input(shape=(self.img_height, self.img_width, 1),
                                  name="ex_img", dtype="float32")
        labels = keras.Input(name='label', shape=(MAX_LEN,), dtype="float32")

        # Convolutional layers
        conv1 = keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1),
                                    padding="same")(input_layer)
        maxpool1 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)

        conv2 = keras.layers.Conv2D(128, kernel_size=(3, 3), strides=(1, 1),
                                    padding="same")(maxpool1)
        maxpool2 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)

        conv3 = keras.layers.Conv2D(256, kernel_size=(3, 3), strides=(1, 1),
                                    padding="same")(maxpool2)
        bn1 = keras.layers.BatchNormalization()(conv3)

        conv4 = keras.layers.Conv2D(256, kernel_size=(3, 3), strides=(1, 1),
                                    padding="same")(bn1)
        maxpool3 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 2))(
            conv4)

        conv5 = keras.layers.Conv2D(512, kernel_size=(3, 3), strides=(1, 1),
                                    padding="same")(maxpool3)
        bn2 = keras.layers.BatchNormalization()(conv5)

        conv6 = keras.layers.Conv2D(512, kernel_size=(3, 3), strides=(1, 1),
                                    padding="same")(bn2)
        maxpool4 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 2))(conv6)

        conv7 = keras.layers.Conv2D(512, kernel_size=(2, 2), strides=(1, 1),
                                    padding="valid")(maxpool4)
        bn3 = keras.layers.BatchNormalization()(conv7)

        # Reshape to prepare for LSTM layers (max-to-sequence)
        conv_output_shape = bn3.shape
        # reshaped_output = keras.layers.Reshape((MAX_LEN, ),)(bn3)
        reshaped_output = keras.layers.Reshape(target_shape=(MAX_LEN, -1))(bn3)

        print(reshaped_output.shape)

        # Add bidirectional LSTM layers
        output = keras.layers.Bidirectional(
            keras.layers.LSTM(256, return_sequences=True))(reshaped_output)
        output = keras.layers.Bidirectional(
            keras.layers.LSTM(256, return_sequences=True))(output)

        # Add dense layer and output layer
        output = keras.layers.Dense(output_shape, activation='softmax')(output)

        # Add CTC layer for calculating CTC loss at each step
        y = CTCLayer()(labels, output)

        # Create the model
        model = keras.Model(
            inputs=[input_layer, labels], outputs=y, name="test_model_v1"
        )

        # Compile the model
        # model.compile(optimizer='adam', loss='categorical_crossentropy')
        model.compile(optimizer='adam')
        # Print the summary of the model
        model.summary()

        return model




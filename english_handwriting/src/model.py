import os
import time
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
# from keras.src.layers import StringLookup
from tensorflow import keras
from util.global_params import BATCH_SIZE


def apply_max_pool(kernel, stride, x):
    return keras.layers.MaxPooling2D(
        pool_size=kernel,
        strides=stride
    )(x)


def apply_cnn(features, kernel, stride, padding, x):
    return keras.layers.Conv2D(
        features,
        kernel,
        strides=stride,
        padding=padding,
        data_format='channels_last'
    )(x)


def apply_BLSTM(hidden, x):
    return keras.layers.Bidirectional(keras.layers.LSTM(hidden, return_sequences=True))(x)


def apply_twice_bn(x):
    out = keras.layers.BatchNormalization(axis=1)(x)
    return keras.layers.BatchNormalization(axis=1)(out)


class CTCLayer(keras.layers.Layer):
    def __int__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = tf.nn.ctc_loss

    def call(self, y_true, y_pred):
        print("y_true " , y_true)
        print("y_pred : " , y_pred)
        batch_len = tf.cast(tf.shape(y_true[0]), dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="float32")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="float32")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)
        return y_pred


class Model:
    # Desired image dimensions
    img_width = 512
    img_height = 64

    def __int__(self):
        pass

    def build_model(self, output_shape):
        input_layer = keras.Input(shape=(self.img_height , self.img_width, 1), name="ex_img", dtype="float32")
        labels = keras.Input(name='label', shape=output_shape, dtype="float32")
        print("label :", labels)
        # 1st CNN layer
        # keras CONV2D filter cause 2 channels
        output = apply_cnn(64, (3, 3), (1, 1), 'same', input_layer )
        # 1st maxpool
        print("frst: ", output.shape)
        output = apply_max_pool((2, 2), (2, 2), output)
        # 2nd CNN
        output = apply_cnn(128, (3, 3), (1, 1), 'same', output)
        output = apply_max_pool((2, 2), (2, 2), output)
        output = apply_cnn(256, (3, 3), (1, 1), 'same', output)
        output = apply_twice_bn(output)
        output = apply_cnn(256, (3, 3), (1, 1), 'same', output)
        output = apply_max_pool((2, 2), (1, 2), output)
        output = apply_cnn(512, (3, 3), (1, 1), 'same', output)
        output = apply_twice_bn(output)
        output = apply_cnn(512, (3, 3), (1, 1), 'same', output)
        output = apply_max_pool((2, 2), (1, 2), output)
        output = apply_cnn(512, (2, 2), (1, 1), 'valid', output)
        output = apply_twice_bn(output)
        # bridge btw CNN and BLSTM
        # sth flattening
        print(output.shape)
        # flatten
        output = keras.layers.Reshape((output.shape[1]*output.shape[2], output.shape[3]), input_shape=output.shape)(output)
        # output = keras.layers.Flatten()(output)
        # output = keras.layers.Dense(512)(output)
        print("flattened : ", output.shape)
        # output = Bidirectional(LSTM(256, return_sequences=True), input_shape=(n_timesteps, 1))
        output = apply_BLSTM(256, output)
        output = apply_BLSTM(256, output)
        print("bidire : ", output.shape)
        output = keras.layers.Dense(output_shape , activation="softmax")(output)
        print("final : ", output.shape)
        output = CTCLayer()(labels, output)

        model = keras.models.Model(
            inputs=input_layer, output=output, name="test_model_v1"
        )

        opt = keras.optimizers.Adam()
        model.compile(optimizer=opt)
        # Transcription layer -Transcription is the process of converting the per-frame predictions made by RNN into a label sequence.
        # translate the per-frame predictions by the recurrent layers into a label sequence
        return model


# if __name__ == '__main__':
#
#     ex_model = Model().build_model()
#     ex_model.summary()
#

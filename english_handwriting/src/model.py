import os
import time
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
# from keras.src.layers import StringLookup
from tensorflow import keras


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
        padding=padding
    )(x)


def apply_BLSTM(hidden, x):
    print(x.shape)
    return keras.layers.Bidirectional(keras.layers.LSTM(hidden, return_sequences=True, dropout=0.25))(x)


def apply_twice_bn(x):
    out = keras.layers.BatchNormalization(axis=1)(x)
    return keras.layers.BatchNormalization(axis=1)(out)


class CTCLayer(keras.layers.Layer):
    def __int__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = tf.nn.ctc_loss

    def call(self, y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true[0]), dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
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

    def build_model(self):
        inp_img = keras.Input(shape=(self.img_width, self.img_height, 1), name="ex_img", dtype="float32")
        labels = keras.Input(name='label', shape=(None,), dtype="float32")

        # 1st CNN layer
        # keras CONV2D filter cause 2 channels
        output = apply_cnn(64, (3, 3), (1, 1), "same", inp_img)
        # 1st maxpool
        output = apply_max_pool((2, 2), (2, 2), output)
        # 2nd CNN
        output = apply_cnn(128, (3, 3), (1, 1), "same", output)
        output = apply_max_pool((2, 2), (2, 2), output)
        output = apply_cnn(256, (3, 3), (1, 1), "same", output)
        output = apply_twice_bn(output)
        output = apply_cnn(256, (3, 3), (1, 1), "same", output)
        output = apply_max_pool((2, 2), (1, 2), output)
        output = apply_cnn(512, (3, 3), (1, 1), "same", output)
        output = apply_twice_bn(output)
        output = apply_cnn(512, (3, 3), (1, 1), "same", output)
        output = apply_max_pool((2, 2), (1, 2), output)
        output = apply_cnn(512, (2, 2), (1, 1), "valid", output)
        output = apply_twice_bn(output)
        # bridge btw CNN and BLSTM
        # sth flattening

        print('output shape', output.shape)
        output = keras.layers.Reshape(target_shape=((self.img_width//6), (self.img_height//6)*256))(output)
        print('output shape', output.shape)
        output = apply_BLSTM(256, output)
        print('output shape', output.shape)
        output = apply_BLSTM(256, output)
        output = keras.layers.Dense(activation="softmax")
        output = CTCLayer(name="ctc_loss")(labels, output)

        model = keras.models.Model(
            inputs=[inp_img, labels], output=output, name="test_model_v1"
        )

        opt = keras.optimizers.Adam()
        model.compile(optimizer=opt)
        model.summary()
        # Transcription layer -Transcription is the process of converting the per-frame predictions made by RNN into a label sequence.
        # translate the per-frame predictions by the recurrent layers into a label sequence
        return model


if __name__ == '__main__':
    ex_model = Model().build_model()
    #ex_model.summary()
    ex_model.fit()
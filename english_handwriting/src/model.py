import tensorflow as tf
from tensorflow import keras
from util.global_params import IMAGE_SIZE, MAX_LEN


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
        # self.loss_fn =  keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)

        self.add_loss(loss)

        # At test time, just return the computed predictions.
        return y_pred


class Model:
    # Desired image dimensions
    img_width = IMAGE_SIZE[1]
    img_height = IMAGE_SIZE[0]

    def __int__(self):
        pass

    def build_model(self, num_classes):
        input_layer = keras.Input(shape=(self.img_height, self.img_width, 1),
                                  name="ex_img", dtype="float32")
        labels = keras.Input(name='label', shape=(MAX_LEN, ), dtype="float32")

        # Convolutional layers

        # 1st CNN layer
        output = apply_cnn(64, (3, 3), (1, 1), 'same', input_layer)
        output = apply_max_pool((2, 2), (2, 2), output)

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

        # bridge btw CNN and BLSTM (max-to-sequence)
        output = keras.layers.Reshape(target_shape=(MAX_LEN, -1))(output)
        print("Reshaped output", output.shape)

        # Add bidirectional LSTM layers
        output = apply_BLSTM(256, output)
        output = apply_BLSTM(256, output)

        # Softmax output layer
        output = keras.layers.Dense(num_classes, activation="softmax")(output)

        # Add CTC layer
        y = CTCLayer()(labels, output)

        # Define the model
        model = keras.Model(
            inputs=[input_layer, labels], outputs=y, name="test_model_v1"
        )

        # Compile the model and return
        opt = keras.optimizers.Adam()
        model.compile(optimizer=opt)

        model.summary()

        return model



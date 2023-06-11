import tensorflow as tf
from tensorflow import keras
from preprocessing import preprocessing
from model import Model


def load_testdata():
    train_batches, val_batches, test_batches, decoder = preprocessing.preprocess(
        True)
    # print(test_batches)
    test_images = []
    test_labels = []
    for batch in train_batches:
        test_images.append(batch["label"])
        test_images.append(batch["image"])
    return test_images, test_labels, decoder, test_batches


def run_test():
    pass


def evaluate(model):
    pass


if __name__ == '__main__':
    img, lbl, decoder, test_batches = load_testdata()
    output_shape = decoder.vocabulary_size() + 2

    model = Model().build_model(output_shape)
    model.load_weights('model_46--191.00')

    #loss, acc = model.evaluate(img, lbl, verbose=2)


    # test_images, test_labels = load_testdata()
    # test_model = keras.models.load_model('saved_model.pb')
    # test_model.summary()
    # loss, acc = test_model.evaluate(test_images, test_labels)

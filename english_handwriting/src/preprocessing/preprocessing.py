import time
from typing import Tuple, Any

import cv2
import pandas as pd
import numpy as np
import imageio.v2 as imageio
import imgaug.augmenters as iaa
from sklearn.model_selection import train_test_split
from tensorflow.python.data.ops.dataset_ops import DatasetV1, DatasetV2

from util.global_params import *
from tensorflow.keras.layers import StringLookup
from util.utils import set_working_dir
from data_agumentation import augment_img


def get_dataframe(path: str) -> pd.DataFrame:
    """
    Extract the image path and the true_label from the IAM lines file and
    create a pandas dataframe
    :param path: str
        Path to the IAM lines file
    :return: pd.DataFrame
        Dataframe containing the image path and the true_label
    """
    data = []
    image_path = None
    label = None

    # in the file, one line contains the image path,
    # the next line the true_label, then an empty line
    # extract the image path and the true_label and create a pandas dataframe
    with open(path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if line:
            if not image_path:
                image_path = line
            else:
                label = line
        else:
            if image_path and label:
                data.append((image_path, label))
                image_path = None
                label = None

    df = pd.DataFrame(data, columns=['image', 'true_label'])

    return df


def split_data(df: pd.DataFrame,
               val_split: float = VAL_SPLIT,
               test_split: float = TEST_SPLIT
               ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the dataframe into training, validation and test sets
    :param df: pd.DataFrame
        Dataframe containing the image path and the true_label
    :param val_split: float
        Validation split
    :param test_split: float
        Test split
    :return: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Training, validation and test sets
    """
    # split the dataframe into training, validation and test sets
    train_df, val_df = train_test_split(
        df, test_size=val_split, random_state=42)

    train_df, test_df = train_test_split(
        train_df, test_size=test_split, random_state=42)

    return train_df, val_df, test_df


def load_images(images_names: list[str]) -> list[np.ndarray]:
    """
    Load and decode the images from the specified filepath
    :param images_names: list[str]
        List of images names
    :return: list[np.ndarray]
        Images as a tf.Tensor list
    """
    images = []

    for image_file in images_names:
        # read the image from the filepath
        file_path = os.path.join(IMAGES_PATH, image_file)
        image = imageio.imread(file_path)
        image = image.astype(np.float32)

        images.append(image)

    return images


def resize_images(images: list[np.ndarray],
                  target_size: tuple[int, int] = IMAGE_SIZE) -> list[tf.Tensor]:
    """
    Resize and pad the images to the specified shape
    :param images: list[np.ndarray]
        List of images as a tf.Tensor
    :param target_size: tuple[int, int]
        Size to resize the images to
        If None, the image is resized to the original size
    :return: list[tf.Tensor]
        Resized images
    """
    (h, w) = (target_size[0], target_size[1])

    resized_images = []

    # resize the images
    for img in images:
        # add the channel dimension
        img = tf.expand_dims(img, axis=-1)

        resized_img = tf.image.resize_with_pad(
            img, target_height=h, target_width=w)
        resized_images.append(resized_img)

    return resized_images


def preprocess_images(images: list[str]) -> list[np.ndarray]:
    """
    Preprocess the images
    :param images: list[str]
        List of images filepaths
    :return: list[tf.Tensor]
        Preprocessed images
    """
    # load the images
    images = load_images(images)

    # resize the images
    images = resize_images(images)

    return images


def augment_training_data(images: list[np.ndarray], labels: list[str]) -> tuple[list[np.ndarray], list[str]]:
    """
    Augment the training data
    :param images: list[np.ndarray]
        List of images
    :param labels: list[str]
        List of labels
    :return: tuple[list[np.ndarray], list[str]]
        Augmented images and labels
    """
    # augment the training data
    # create a pandas dataframe from the images and labels
    df = pd.DataFrame({'image': images,
                       'true_label': labels})

    augmented_images = []
    augmented_labels = []

    for image, label in zip(images, labels):
        # augment the image
        augmented_image = augment_img(image)
        if isinstance(augmented_image, dict):
            augmented_image = augmented_image['image']

        if not np.array_equal(image, augmented_image):
            augmented_images.append(augmented_image)
            augmented_labels.append(label)

    augmented_df = pd.DataFrame({'image': augmented_images,
                                 'true_label': augmented_labels})

    new_df = pd.concat([df, augmented_df], ignore_index=True)

    return new_df['image'].values, new_df['true_label'].values


def get_vocabulary_from_labels(labels: list[str]) -> tuple[list[str], int]:
    """
    Get the vocabulary from the labels and the maximum true_label length
    :param labels: list[str]
        List of labels
    :return: tuple[list[str], int]
        Vocabulary and maximum true_label length
    """

    # get the vocabulary and the max len from the training labels
    vocab = set()
    max_label_len = 0
    for label in labels:
        vocab.update(label)
        if len(label) > max_label_len:
            max_label_len = len(label)

    vocab = sorted(vocab)
    max_label_len = max(max_label_len, MAX_LEN)

    return vocab, max_label_len


def get_encoding(vocab: list[str]) -> tuple[StringLookup, StringLookup]:
    """
    Get the encoding and decoding lookup tables
    :param vocab: list[str]
        Vocabulary
    :return: tuple[tf.lookup.StringLookup, tf.lookup.StringLookup]
        Encoding and decoding lookup tables
    """

    # Mapping characters to integers.
    char_to_num = StringLookup(vocabulary=list(vocab), mask_token=None)

    # Mapping integers back to original characters.
    num_to_char = StringLookup(
        vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
    )

    return char_to_num, num_to_char


def encode_labels(labels: list[str], encoder: StringLookup,
                  max_label_len: int) -> list[tf.Tensor]:
    """
    Encode the labels and pad them to the same length
    :param labels: list[str]
        List of labels
    :param encoder: tf.lookup.StringLookup
        Encoder
    :param max_label_len: int
        Maximum true_label length
    :return: list[tf.Tensor]
        Encoded and padded labels
    """
    padded_labels = []

    for label in labels:
        # encode the true_label
        encoded_label = encoder(
            tf.strings.unicode_split(label, input_encoding="UTF-8"))

        # pad the true_label to the maximum length
        length = tf.shape(encoded_label)[0]
        pad_amount = max_label_len - length
        encoded_label = tf.pad(
            encoded_label, paddings=[[0, pad_amount]],
            constant_values=PADDING_TOKEN)

        padded_labels.append(encoded_label)

    return padded_labels


def get_decoder(decoder_vocab: list[str]) -> StringLookup:
    """
    Decode the true_label
    :param decoder_vocab: list[str]
        Decoder vocabulary
    :return: str
        Decoded true_label
    """
    # get the decoder
    decoder = StringLookup(vocabulary=decoder_vocab,
                           mask_token=None, invert=True)

    return decoder


def generator(images: list[np.ndarray], labels: list[tf.Tensor]) -> dict[
    str, tf.Tensor]:
    """
    Generator for the tf.data.Dataset
    :param images: list[tf.Tensor]
        List of images
    :param labels: list[tf.Tensor]
        List of labels
    :return: dict[tf.Tensor, tf.Tensor]
        Image and true_label
    """
    for img, label in zip(images, labels):
        yield {"image": img, "true_label": label}


def save_dataset(dataset: tf.data.Dataset,
                 name: str, path: str = PREPROCESSED_DATA_PATH):
    """
    Save the dataset to the specified path as a csv file
    :param dataset: tf.data.Dataset
        Dataset to save
    :param name: str
        Name of the dataset
    :param path: str
        Path to save the dataset to
    """
    # create dir if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)

    file_path = os.path.join(path, name + '.csv')

    # create a dataframe from the dataset
    df = pd.DataFrame(dataset.as_numpy_iterator(),
                      columns=['image', 'true_label'])

    # save the dataframe to a csv file
    df.to_csv(file_path, index=False)


def get_data_path(folder: str) -> str:
    """
    Get the path to the preprocessed data
    :param folder: str
        Folder name
    :return: str
        Path to the preprocessed data
    """
    return os.path.join(PREPROCESSED_DATA_PATH, folder)


def load_vocab(path: str) -> list[str]:
    """
    Load the decoder
    :param path: str
        Path to the decoder
    :return: list[str]
        Decoder
    """
    vocab = []
    with open(path, 'r') as f:
        for line in f:
            symbol = line.strip()
            if symbol == '':
                symbol = ' '
            vocab.append(symbol)
    return vocab


def save_decoder_vocab(decoder: list[str], path: str):
    """
    Save the vocabulary for the decoder
    :param decoder: list[str]
        Decoder
    :param path: str
        Path to save the decoder to
    """
    with open(path, 'w') as f:
        for char in decoder:
            f.write(char + '\n')


def preprocess_data(print_progress: bool = False):
    """
    Call all the preprocessing functions to preprocess the data
    """
    time_start = time.time()
    if print_progress:
        print('Task\t\t\tTime elapsed (seconds)')
        print('----------------------------------')

    # read the lines file
    df = get_dataframe(IAM_LINES_PATH)
    if print_progress:
        print('Read lines file\t',
              time.time() - time_start)

    # split the dataset
    train_df, val_df, test_df = split_data(df)
    if print_progress:
        print('Split dataset\t',
              time.time() - time_start)

    # get the vocabulary from the training labels
    vocab, max_label_len = get_vocabulary_from_labels(train_df['true_label'].values)
    if print_progress:
        print('Get vocabulary\t',
              time.time() - time_start)

    # preprocess/load the images
    train_imgs = load_images(train_df['image'].values)
    val_imgs = preprocess_images(val_df['image'].values)
    test_imgs = preprocess_images(test_df['image'].values)
    if print_progress:
        print('Encode images\t',
              time.time() - time_start)

    # encode the labels
    encoder, decoder = get_encoding(vocab)

    # augment the training data
    train_imgs, train_labels = augment_training_data(
        train_imgs, train_df['true_label'].values)
    train_imgs = resize_images(train_imgs)
    if print_progress:
        print('Augment training data\t',
              time.time() - time_start)

    encoded_train_labels = encode_labels(
        train_labels, encoder, max_label_len)
    encoded_val_labels = encode_labels(
        val_df['true_label'].values, encoder, max_label_len)
    encoded_test_labels = encode_labels(
        test_df['true_label'].values, encoder, max_label_len)
    if print_progress:
        print('Encode labels\t',
              time.time() - time_start, "seconds")

    train_data = tf.data.Dataset.from_generator(
        lambda: generator(train_imgs, encoded_train_labels),
        output_signature=(
            {
                'image': tf.TensorSpec(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 1), dtype=tf.float32),
                'true_label': tf.TensorSpec(shape=(MAX_LEN,), dtype=tf.int64)
            }
        )
    )

    val_data = tf.data.Dataset.from_generator(
        lambda: generator(val_imgs, encoded_val_labels),
        output_signature=(
            {
                'image': tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32),
                'true_label': tf.TensorSpec(shape=(None,), dtype=tf.int64)
            }
        )
    )

    test_data = tf.data.Dataset.from_generator(
        lambda: generator(test_imgs, encoded_test_labels),
        output_signature=(
            {
                'image': tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32),
                'true_label': tf.TensorSpec(shape=(None,), dtype=tf.int64)
            }
        )
    )

    if print_progress:
        print('Create dataset\t',
              time.time() - time_start, "seconds")

    return train_data, val_data, test_data, decoder


def preprocess(print_progress=False):
    """
    Preprocess the data
    """
    if print_progress:
        print('Preprocessing data')

    train_path = get_data_path('train')
    val_path = get_data_path('val')
    test_path = get_data_path('test')
    vocab_path = get_data_path('vocab.txt')

    # check if the dataset has already been preprocessed
    if os.path.exists(train_path) and os.path.exists(val_path) and \
            os.path.exists(test_path) and os.path.exists(vocab_path):

        if print_progress:
            print('Loading preprocessed data')

        train_data = tf.data.Dataset.load(train_path)
        val_data = tf.data.Dataset.load(val_path)
        test_data = tf.data.Dataset.load(test_path)

        # load the decoder
        decoder_vocab = load_vocab(vocab_path)
        decoder = get_decoder(decoder_vocab)

    else:
        train_data, val_data, test_data, decoder = \
            preprocess_data(print_progress=print_progress)

        # save the datasets

        # create dir if it doesn't exist
        if not os.path.exists(PREPROCESSED_DATA_PATH):
            os.makedirs(PREPROCESSED_DATA_PATH)

        train_data.save(train_path)
        val_data.save(val_path)
        test_data.save(test_path)

        # save the decoder
        save_decoder_vocab(decoder.get_vocabulary(), vocab_path)

    if print_progress:
        print('Creating batches')

    train_batches = train_data.batch(BATCH_SIZE).cache().prefetch(
        buffer_size=AUTOTUNE)
    val_batches = val_data.batch(BATCH_SIZE).cache().prefetch(
        buffer_size=AUTOTUNE)
    test_batches = test_data.batch(BATCH_SIZE).cache().prefetch(
        buffer_size=AUTOTUNE)

    return train_batches, val_batches, test_batches, decoder


def preprocess_test_data(path: str) -> tuple[DatasetV1 | DatasetV2, StringLookup]:
    """
    Preprocess the test data
    """
    test_df = get_dataframe(path)
    vocab_path = get_data_path('vocab.txt')

    test_imgs = preprocess_images(test_df['image'].values)

    # encode the labels
    vocab = load_vocab(vocab_path)
    encoder, decoder = get_encoding(vocab)
    encoded_test_labels = encode_labels(
        test_df['true_label'].values, encoder, MAX_LEN)

    test_data = tf.data.Dataset.from_generator(
        lambda: generator(test_imgs, encoded_test_labels),
        output_signature=(
            {
                'image': tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32),
                'true_label': tf.TensorSpec(shape=(None,), dtype=tf.int64)
            }
        )
    )

    return test_data, decoder


if __name__ == '__main__':
    # change working dir to root of project
    set_working_dir(os.path.abspath(__file__))
    preprocess(print_progress=True)

import time
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.src.layers import StringLookup
from util.global_params import *


def get_dataframe(path: str) -> pd.DataFrame:
    """
    Extract the image path and the label from the IAM lines file and
    create a pandas dataframe
    :param path: str
        Path to the IAM lines file
    :return: pd.DataFrame
        Dataframe containing the image path and the label
    """
    data = []
    image_path = None
    label = None

    # in the file, one line contains the image path,
    # the next line the label, then an empty line
    # extract the image path and the label and create a pandas dataframe
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

    df = pd.DataFrame(data, columns=['image', 'label'])

    return df


def split_data(df: pd.DataFrame,
               val_split: float = VAL_SPLIT,
               test_split: float = TEST_SPLIT
               ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the dataframe into training, validation and test sets
    :param df: pd.DataFrame
        Dataframe containing the image path and the label
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


def load_images(images_names: list[str]) -> list[tf.Tensor]:
    """
    Load and decode the images from the specified filepath
    :param images_names: list[str]
        List of images names
    :return: list[tf.Tensor]
        Images as a tf.Tensor list
    """
    images = []

    for image_file in images_names:
        # read the image from the filepath
        file_path = os.path.join(IMAGES_PATH, image_file)
        image = tf.io.read_file(file_path)
        image = tf.image.decode_image(image, channels=1)
        image = tf.cast(image, tf.float32) / 255.0

        # binarize the image
        # image = tf.where(image > binarize_threshold, 1, 0)

        images.append(image)

    return images


def resize_images(images: list[tf.Tensor],
                  size: tuple[int, int] = IMAGE_SIZE) -> list[tf.Tensor]:
    """
    Resize the images to the specified shape
    :param images: list[tf.Tensor]
        List of images as a tf.Tensor
    :param size: tuple[int, int]
        Size to resize the images to
        If None, the image is resized to the original size
    :return: list[tf.Tensor]
        Resized images
    """
    # if shape is not specified, use the maximum width and height of the images
    if size is None:
        size = np.max([list(img.shape) for img in images], axis=0)

    (h, w) = (size[0], size[1])

    resized_images = []

    # resize the images
    for img in images:
        resized_img = tf.image.resize_with_pad(
            img, target_height=h, target_width=w)
        resized_images.append(resized_img)

    return resized_images


def preprocess_images(images: list[str]) -> list[tf.Tensor]:
    """
    Preprocess the images
    :param images: list[str]
        List of images filepaths
    :return: list[tf.Tensor]
        Preprocessed images
    """
    # load the images and binarize them
    images = load_images(images)

    # resize the images
    images = resize_images(images)

    return images


def get_vocabulary(labels: list[str]) -> tuple[list[str], int]:
    """
    Get the vocabulary from the labels and the maximum label length
    :param labels: list[str]
        List of labels
    :return: tuple[list[str], int]
        Vocabulary and maximum label length
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
        Maximum label length
    :return: list[tf.Tensor]
        Encoded and padded labels
    """
    padded_labels = []

    for label in labels:
        # encode the label
        encoded_label = encoder(
            tf.strings.unicode_split(label, input_encoding="UTF-8"))

        # pad the label to the maximum length
        length = tf.shape(encoded_label)[0]
        pad_amount = max_label_len - length
        encoded_label = tf.pad(
            encoded_label, paddings=[[0, pad_amount]],
            constant_values=PADDING_TOKEN)

        padded_labels.append(encoded_label)

    return padded_labels


def get_decoder(decoder_vocab: list[str]) -> StringLookup:
    """
    Decode the label
    :param decoder_vocab: list[str]
        Decoder vocabulary
    :return: str
        Decoded label
    """
    # get the decoder
    decoder = StringLookup(vocabulary=decoder_vocab,
                           mask_token=None, invert=True)

    return decoder


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
                      columns=['image', 'label'])

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


def load_decoder(path: str) -> list[str]:
    """
    Load the decoder
    :param path: str
        Path to the decoder
    :return: list[str]
        Decoder
    """
    decoder = []
    with open(path, 'r') as f:
        for line in f:
            decoder.append(line.strip())

    return decoder


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
        print('Preprocessing data...')
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
    vocab, max_label_len = get_vocabulary(train_df['label'].values)
    if print_progress:
        print('Get vocabulary\t',
              time.time() - time_start)

    # preprocess/encode the images
    train_imgs = preprocess_images(train_df['image'].values)
    val_imgs = preprocess_images(val_df['image'].values)
    test_imgs = preprocess_images(test_df['image'].values)
    if print_progress:
        print('Encode images\t',
              time.time() - time_start)

    # encode the labels
    encoder, decoder = get_encoding(vocab)

    encoded_train_labels = encode_labels(
        train_df['label'].values, encoder, max_label_len)
    encoded_val_labels = encode_labels(
        val_df['label'].values, encoder, max_label_len)
    encoded_test_labels = encode_labels(
        test_df['label'].values, encoder, max_label_len)
    if print_progress:
        print('Encode labels\t',
              time.time() - time_start, "seconds")

    # create a tf.data.Dataset and save it as a csv file
    train_data = tf.data.Dataset.from_tensor_slices(
        (train_imgs, encoded_train_labels))
    val_data = tf.data.Dataset.from_tensor_slices(
        (val_imgs, encoded_val_labels))
    test_data = tf.data.Dataset.from_tensor_slices(
        (test_imgs, encoded_test_labels))
    if print_progress:
        print('Create dataset\t',
              time.time() - time_start, "seconds")

    return train_data, val_data, test_data, decoder


def preprocess():
    """
    Preprocess the data
    """
    train_path = get_data_path('train')
    val_path = get_data_path('val')
    test_path = get_data_path('test')
    vocab_path = get_data_path('vocab.txt')

    # check if the dataset has already been preprocessed
    if os.path.exists(train_path) and os.path.exists(val_path) and \
            os.path.exists(test_path) and os.path.exists(vocab_path):

        train_data = tf.data.Dataset.load(train_path)
        val_data = tf.data.Dataset.load(val_path)
        test_data = tf.data.Dataset.load(test_path)

        # load the decoder
        decoder_vocab = load_decoder(vocab_path)
        decoder = get_decoder(decoder_vocab)

    else:
        train_data, val_data, test_data, decoder = \
            preprocess_data(print_progress=True)

        # save the datasets

        # create dir if it doesn't exist
        if not os.path.exists(PREPROCESSED_DATA_PATH):
            os.makedirs(PREPROCESSED_DATA_PATH)

        train_data.save(train_path)
        val_data.save(val_path)
        test_data.save(test_path)

        # save the decoder
        save_decoder_vocab(decoder.get_vocabulary(), vocab_path)

    # create batches
    train_batches = train_data.batch(BATCH_SIZE).cache().prefetch(
        buffer_size=AUTOTUNE)
    val_batches = val_data.batch(BATCH_SIZE).cache().prefetch(
        buffer_size=AUTOTUNE)
    test_batches = test_data.batch(BATCH_SIZE).cache().prefetch(
        buffer_size=AUTOTUNE)

    return train_batches, val_batches, test_batches, decoder


if __name__ == '__main__':
    preprocess()

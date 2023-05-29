import random
import cv2
import os
import numpy as np
from glob import glob
from PIL import Image, ImageOps

from generate_data.generator_data import create_image
from generate_data.text_generator_ngram import generator
from typing import Union, Tuple
import string
import imgaug.augmenters as iaa
from Augmentation.LineAugmentation import rotate_several_by_degree

# should be based on N-gram probability distribution
# FOLDER = 'train'
FOLDER = 'output'
WORD_LENGTH = 10
TEXT_LENGTH = 100 * np.random.randint(1, 5, size=1)[0]
# text_len = 10
# SCRIPT_SIZE = 608
NGRAM_SIZE = 4
Box = [float, float, float, float]
WIDTH = 640
HEIGHT = 640
PADDING = 10 * np.random.randint(10, size=1)[0]
WHITESPACE = 15
PATH = os.getcwd()
SCRIPT_NAME = 'test3'
# LETTERS_FOLDER = os.path.join('C:/Users/admin/Documents/Uni/Master/4.Sem/HWR', 'symbols')

# LETTERS_FOLDER = '../data/preprocessed_images/symbols'
SCRIPT_NAME = 'test'
LETTERS_FOLDER = os.path.join('..\preprocess', 'output', 'symbols')

DATA_FOLDER = 'datasets'
labels = {'Alef': 0,
          'Ayin': 1,
          'Bet': 2,
          'Dalet': 3,
          'Gimel': 4,
          'He': 5,
          'Het': 6,
          'Kaf': 7,
          'Kaf-final': 8,
          'Lamed': 9,
          'Mem': 10,
          'Mem-medial': 11,
          'Nun-final': 12,
          'Nun-medial': 13,
          'Pe': 14,
          'Pe-final': 15,
          'Qof': 16,
          'Resh': 17,
          'Samekh': 18,
          'Shin': 19,
          'Taw': 20,
          'Tet': 21,
          'Tsadi-final': 22,
          'Tsadi-medial': 23,
          'Waw': 24,
          'Yod': 25,
          'Zayin': 26}


def resize_data(image: Image.Image, width: int, height: int) -> Tuple[
    Image.Image, Tuple[float, float]]:
    """
    Normalizes images size to WIDTH and HEIGHT

    Parameters
    ----------
        image (numpy.array): image to procell
        width (int): Width dimension of image in pixels.
        height (int): Height dimension of image in pixels.

    ----------
    Returns:
        resized_image (PIL.Image.Image) : Resized image
        resized_shape (float, float): Image shape
    """

    resized_image = image.resize((width, height))
    resized_shape = resized_image.size

    return resized_image, resized_shape


def save_coco_label(file: str, label_class: str, points: Box, path: str, folder: str):
    """
    Saves the label of the image in coco format: classs, x_c, y_c, w, h

    Parameters
    ----------
        file (str): Name of label (same as name of image)
        points (Box): x_c, y_c, w, h
        path (str): Path where image is saved
    """
    x_c = points[0]
    y_c = points[1]
    w = points[2]
    h = points[3]
    label_class = labels[label_class]
    label = '{} {} {} {} {}'.format(label_class, x_c, y_c, w, h).replace('"', '')
    file = str(file) + '.txt'
    # with open(os.path.join(DATA_FOLDER, "labels", folder, str(file)), 'a') as f:
    with open(os.path.join("labels", str(file)), 'a') as f:
        f.write(label)
        f.write("\n")
        f.close()


def load_class_images(folder):
    """
    Load the images in a list from the specified folder
    Name of folder should be the name of each letter
    Parameters
    ----------
        folder (str): Name of folder to load images from
    """

    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_UNCHANGED)
        if img is not None:
            images.append(img)
    return images


def load_classes(folders):
    """
    Make a dict with {key: 'letter_name', value: list[imgs]}

    Parameters
    ----------
    """

    classes = {}
    for folder_class in folders:
        class_data = load_class_images(folder_class)
        # # the last directory in the path folder_class
        class_name = os.path.basename(os.path.normpath(folder_class))
        classes[class_name] = class_data
    return classes


# TODO: might use this function in the final pipeline
# def padding(img, expected_size):
#     """
#     Make a dict with {key: 'letter_name', value: list[imgs]}
#
#     Parameters
#     ----------
#     """
#     desired_size = expected_size
#     delta_width = desired_size[0] - img.size[0]
#     delta_height = desired_size[1] - img.size[1]
#     pad_width = delta_width // 2
#     pad_height = delta_height // 2
#     padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
#     return ImageOps.expand(img, padding)


def draw_boxes(img, x, y, w, h):
    dh, dw, _ = np.asarray(img).shape
    l = int((x - w / 2) * dw)
    r = int((x + w / 2) * dw)
    t = int((y - h / 2) * dh)
    b = int((y + h / 2) * dh)

    if l < 0:
        l = 0
    if r > dw - 1:
        r = dw - 1
    if t < 0:
        t = 0
    if b > dh - 1:
        b = dh - 1

    image = cv2.rectangle(np.asarray(img), (l, t), (r, b), (255, 0, 0), 1)
    cv2.imshow('image', image)
    # Displaying the image
    cv2.waitKey(0)


def stitch(images, text, folder, script_name):
    """
    Create a new image for the script with fixed size. Start adding from right to left
    the selected letters from the generated text. When PADDING is reached in width (x-axis), make
    a new line by updating the y-axis.

    Parameters
    ----------
    """
    images_type = []
    for i in images:
        i = Image.fromarray(i)
        images_type.append(i)
    widths, heights = zip(*(i.size for i in images_type))
    max_height = max(heights)
    new_im = Image.new('RGB', (WIDTH, HEIGHT), color="white")

    x_offset = PADDING * np.random.randint(1, 4, size=1)[0]
    start_offset = x_offset
    y_offset = PADDING * np.random.randint(1, 2, size=1)[0]
    line_calc = 0
    col_calc = 0

    for idx, im in enumerate(images_type):
        if im.size[0] > 1 and im.size[1] > 1:
            col_calc += 1
            # im = im.resize((int(im.size[0] / 2), int(im.size[1] / 2)))
            im = im.resize((int(im.size[0] / 2), int(im.size[1] / 2)))
        if new_im.size[0] - (x_offset + im.size[0]) <= start_offset:
            line_calc += 1
            print('col calc', np.abs(col_calc-(len(images_type)-idx)))
            #if np.abs(col_calc-(len(images_type)-idx))<=int(len(images_type)/2):

            print('line calc', line_calc)
            col_calc = 0
            # y_offset = y_offset + int(max_height / 2)
            y_offset = y_offset + int(max_height / 2)
            x_offset = start_offset
        else:
            #  CHANGE Y OFFSET BASED ON (NEW_IMG.HEIGTH - IM.HEIGTH) / 2
            y_offset = y_offset
            # (x_offset, 0) = upper left corner of the canvas where to paste
        # cropping = np.random.randint(5, 8, size=1)[0]
        cropping = np.random.randint(5, 6, size=1)[0]
        if im.size[0] > 10:
            im = im.crop((cropping, 0, im.size[0] - cropping, im.size[1]))
        new_im.paste(im, (new_im.size[0] - (x_offset + im.size[0]), y_offset))
        w = im.size[0]
        h = im.size[1]
        x_c = (new_im.size[0] - (x_offset + w / 2)) / WIDTH
        y_c = (y_offset + h / 2) / HEIGHT
        w = w / WIDTH
        h = h / HEIGHT
        box = [x_c, y_c, w, h]
        # label = text[idx]
        label = text[idx if idx <= 0 else idx - 1]
        # skip punctuations, do not save their labels, yolo should not learn them
        if label not in string.punctuation:
            save_coco_label(script_name, label, box, PATH, folder)
            # uncomment to draw boxes
            # draw_boxes(new_im, x_c, y_c, w, h)
        # uncomment to see the process of stitching
        # new_im.show()
        #  slide the upper left corner for pasting next image next iter
        x_offset = x_offset + im.size[0]

    # augment the final image
    new_im = transform_scroll(new_im)

    new_im.save(
        # os.path.join(DATA_FOLDER, 'images', folder, script_name + '.png'))
        os.path.join('datasets', 'images', script_name + '.png'))


def get_random_param_values():
    """
    Get random values for the parameters of the image augmentation.
    The same values are used for all the letters in a script.
    """
    # random values for the parameters of the image augmentation
    rotation = (-15, 15)
    shear = (-15, 15)
    gauss_blur_sigma = (0, 2.0)
    crop = (-0.2, 0.2)
    elastic_alpha = (0, 20)
    elastic_sigma = (4, 6)

    # for each parameter, select a random value from the range
    rotation = random.uniform(*rotation)
    shear = random.uniform(*shear)
    gauss_blur_sigma = random.uniform(*gauss_blur_sigma)
    crop = random.uniform(*crop)
    elastic_alpha = random.uniform(*elastic_alpha)
    elastic_sigma = random.uniform(*elastic_sigma)

    return rotation, shear, gauss_blur_sigma, crop, elastic_alpha, elastic_sigma


def pil_to_ndarray(image) -> np.ndarray:
    """
    Convert a PIL image to a numpy array.
    """
    image = np.asarray(image)
    return image


def ndarray_to_pil(image: np.ndarray) -> Image:
    """
    Convert a numpy array to a PIL image.
    """
    image = Image.fromarray(image)
    return image


def transform_letter(image: np.ndarray, rotation: float, shear: float,
                     gauss_blur_sigma: float, crop: float, elastic_alpha: float,
                     elastic_sigma: float) -> np.ndarray:
    """
    Apply a series of transformations to the letter image and return the augmented image.
    """
    image = pil_to_ndarray(image)

    # the same augmentation applied to be applied to each letter image in a script
    aug = iaa.Sequential([
        iaa.Affine(rotate=rotation, shear=shear, mode='constant', cval=255),
        iaa.GaussianBlur(sigma=gauss_blur_sigma),
        iaa.KeepSizeByResize(
            iaa.CropAndPad(percent=crop, pad_mode='constant', pad_cval=255)),
        iaa.Pad(px=2, pad_mode='constant', pad_cval=255),
        iaa.ElasticTransformation(alpha=elastic_alpha, sigma=elastic_sigma),
        iaa.size.Crop(px=2)
    ])

    aug_image = aug(image=image)

    # apply a random harser crop (up to 50% of the image)
    aug_crop = iaa.KeepSizeByResize(
        iaa.CropAndPad(percent=(-0.5, 0.5), pad_mode='constant', pad_cval=255))

    # with a small probability, crop the image
    if np.random.random_sample() < 0.1:
        aug_image = aug_crop(image=aug_image)

    return aug_image


def transform_scroll(image) -> Image:
    """
    Apply a series of transformations to the script image and return the augmented image.
    """
    image = pil_to_ndarray(image)

    # random cutouts with white pixels to simulate noise/scroll erosion
    cutout = iaa.Cutout(nb_iterations=(50, 100), size=(0.01, 0.05),
                        squared=False, fill_mode="constant", cval=255)

    aug_image = cutout(image=image)
    aug_image = ndarray_to_pil(aug_image)

    return aug_image


def sample_text_generator(text_len, ngram_size):
    """
    Main function calling all the other helper function.
    Generate text, transform it into script.

    Parameters
    ----------
    """
    class_names = []
    for path in glob(f'{LETTERS_FOLDER}/*/'):
        class_names.append(path)

    # class_names = glob(LETTERS_FOLDER + os.sep + "*", recursive=False)
    images = load_classes(class_names)
    text = generator(text_len, ngram_size)

    # remove all dots from text to prepare text for labelling
    text = text.split(" ")
    script = []

    # set the parameters for the image augmentation
    params = get_random_param_values()
    for idx, letter in enumerate(text):
        if letter not in string.punctuation and letter != '':
            random_sample_idx = np.random.choice(len(images[letter]), 1)[0]
            random_param = np.random.random_sample()
            if random_param < 0.5:
                random_sample = create_image(letter, (64, 69)).convert('1')
                random_sample = np.array(random_sample)
            else:
                random_sample = images[letter][random_sample_idx]
            # add some transformations to the image (letter)
            random_sample = transform_letter(random_sample, *params)
        else:
            end_token = np.zeros(
                [WHITESPACE, np.random.randint(1, 3, size=1)[0] * WHITESPACE],
                dtype=np.uint8)
            end_token.fill(255)
            random_sample = end_token
        script.append(random_sample)
    # script = np.array(script)
    return script, text


def generate_sample(folder, script_name, text_length=TEXT_LENGTH):
    class_names = glob(LETTERS_FOLDER + os.sep + "*", recursive=False)

    # images = load_classes(class_names)
    script, text = sample_text_generator(text_length, NGRAM_SIZE)

    # THIS IS VYVY'S PART
    # for im in script:
    #       print(im.shape)
    '''   for im in script:
        if im.shape[0]>20:'''

    # script = rotate_several_by_degree([im for im in script if im.shape[0] > 20])

    stitch(script, text, folder, script_name)


# for i in range(10):
#     generate_sample(FOLDER, f"sample{i}_with_crop_and_cutout")

generate_sample(FOLDER, "test")

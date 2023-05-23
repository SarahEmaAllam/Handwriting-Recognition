import cv2
import os
import numpy as np
from glob import glob
from PIL import Image, ImageOps
from text_generator_ngram import generator
from typing import Union, Tuple
import string
import imgaug.augmenters as iaa
from LineAugmentation import rotate

# should be based on N-gram probability distribution
FOLDER = 'train'
WORD_LENGTH = 10
TEXT_LENGTH = 100 * np.random.randint(1, 5, size=1)[0]
# TEXT_LENGTH = 10
# SCRIPT_SIZE = 608
NGRAM_SIZE = 4
Box = [float, float, float, float]
WIDTH = 640
HEIGHT = 640
PADDING = 10 * np.random.randint(5, size=1)[0]
WHITESPACE = 10
PATH = os.getcwd()
SCRIPT_NAME = 'test3'
LETTERS_FOLDER = os.path.join('..','preprocess', 'output', 'symbols')
# LETTERS_FOLDER = '../data/preprocessed_images/symbols'


def resize_data(image: Image.Image, width: float, height: float) -> Tuple[
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


def save_coco_label(file: str, label: str, points: Box, path: str):
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
    label = '{} {} {} {} {}'.format(label, x_c, y_c, w, h).replace('"', '')
    file = str(file) + '.txt'
    with open(os.path.join("..","data", "labels", FOLDER, str(file)), 'a') as f:
        f.write(label)
        f.write("\n")
        f.close()


def load_class_images(folder):
    """
    Load the images in a list from the specified folder
    Name of folder should be name of each letter
    Parameters
    ----------
        folder (str): Name of folder to load images from
    """

    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename),cv2.IMREAD_UNCHANGED)
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
        class_name = folder_class.split(os.sep)[4]
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

    image = cv2.rectangle(np.asarray(img), (l, t), (r, b), (255, 0, 0) , 1)
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
    for idx, im in enumerate(images_type):
        print("before ", im.size)
        if im.size[0] > 1 and im.size[1] > 1:
            im = im.resize((int(im.size[0] / 2), int(im.size[1] / 2)))
            print(im.size)
        if new_im.size[0] - (x_offset + im.size[0]) <= start_offset:
            y_offset = y_offset + int(max_height / 2)
            x_offset = start_offset
        else:
            #  CHANGE Y OFFSET BASED ON (NEW_IMG.HEIGTH - IM.HEIGTH) / 2
            y_offset = y_offset
            # (x_offset, 0) = upper left corner of the canvas where to paste
        cropping = np.random.randint(7 , 10, size=1)[0]
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
        label = text[idx]
        # skip punctuations, do not save their labels, yolo should not learn them
        if label not in string.punctuation:
            save_coco_label(script_name, label, box, PATH)
            # uncomment ot draw boxes
            # draw_boxes(new_im, x_c, y_c, w, h)
        # uncomment to see the process of stitching
        # new_im.show()
        #  slide the upper left corner for pasting next image next iter
        x_offset = x_offset + im.size[0]
    new_im.save(os.path.join('..','data', 'images', folder, script_name + '.png'))

def transform_letter(image: Image.Image) -> Image.Image:
    """
    Apply a series of transformations to the image and return the augmented image.
    """
    aug = iaa.Sequential([
        iaa.Affine(rotate=(-15, 15), shear=(-15, 15), mode='constant', cval=255),
        iaa.GaussianBlur(sigma=(0, 2.0)),
        iaa.KeepSizeByResize(
            iaa.CropAndPad(percent=(-0.1, 0.1), pad_mode='constant', pad_cval=255)),
        iaa.Pad(px=2, pad_mode='constant', pad_cval=255),
        iaa.ElasticTransformation(alpha=(0, 20), sigma=(4, 6)),
        iaa.size.Crop(px=2)
    ])

    return aug(image=image)


def sample_text_generator(TEXT_LENGTH, NGRAM_SIZE):
    """
    Main function calling all the other helper function.
    Generate text, transform it into script.

    Parameters
    ----------
    """
    class_names = glob(LETTERS_FOLDER + os.sep+"*", recursive=False)
    images = load_classes(class_names)
    text = generator(TEXT_LENGTH, NGRAM_SIZE)

    # remove all dots from text to prepare text for labelling
    text = text.split(" ")
    script = []

    for letter in text:
        if letter not in string.punctuation and letter != '':
            random_sample_idx = np.random.choice(len(images[letter]), 1)[0]
            random_sample = images[letter][random_sample_idx]
            # add some transformations to the image (letter)
            random_sample = transform_letter(random_sample)
        else:
            end_token = np.zeros(
                [WHITESPACE, np.random.randint(3, size=1)[0] * WHITESPACE],
                dtype=np.uint8)
            end_token.fill(255)
            random_sample = end_token

        script.append(random_sample)

    script = np.array(script)
    return script, text

def generate_sample(folder, script_name):
    class_names = glob(LETTERS_FOLDER + os.sep+ "*", recursive=False)
    print(class_names)

    # images = load_classes(class_names)
    script, text = sample_text_generator(TEXT_LENGTH, NGRAM_SIZE)

    # THIS IS VYVY'S PART
    # for im in script:
    #     print(im.shape)
    # rotate([im for im in script if im.shape[0] > 20])

    stitch(script, text, folder, script_name)

generate_sample(FOLDER, SCRIPT_NAME)
import math

import cv2
import os
import numpy as np
from glob import glob
from PIL import Image, ImageOps
from text_generator_ngram import generator
from typing import Union, Tuple
import string
# should be based on N-gram probability distribution
WORD_LENGTH = 10
TEXT_LENGTH= 200 *  np.random.randint(5, size=1)[0]
# TEXT_LENGTH= 10
SCRIPT_SIZE = 608
NGRAM_SIZE = 4
Box = [float, float, float, float]
WIDTH = 1700
HEIGHT = 1700
PADDING = 50 *  np.random.randint(5, size=1)[0]
WHITESPACE = 30
PATH = os.getcwd()
SCRIPT_NAME = 'test'
LETTERS_FOLDER = os.path.join('preprocess','output', 'symbols')


def resize_data(image: Image.Image, width: float, height: float) -> Tuple[Image.Image, Tuple[float, float]]:
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

def save_coco_label(file: str, label:str, points: Box, path: str):
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
    with open(os.path.join(str(path),"labels",str(file)), 'a') as f:
        f.write(label)
        f.write("\n")
        f.close()

# def preprocess(img: Image.Image, annotation: np.ndarray):
#     """
#     Resizes and reformats the images to Coco standard
#     Prepares images for the yolov5 model
#
#     Parameters
#     ----------
#         img: Image
#         annotation: dict
#
#     Returns
#     -------
#         images (numpy.ndarray): numpy array of loaded images
#         points (Box): points for the boundary box
#         props_shape (prop_x, prop_y): keeps track of the proportion between old image size and new one
#     """
#
#     init_shape = img.size
#     img, resized_shape = resize_data(img, WIDTH, HEIGHT)
#     prop_shape = (resized_shape[0] / init_shape[0], resized_shape[1] / init_shape[1])
#
#     # points = Google_to_Coco(annotation, prop_shape)
#     return img, points, prop_shape

def stitch(imagefiles):
    for filename in imagefiles:
        img = cv2.imread(filename)
        images.append(img)
    return images


def load_class_images(folder):
    images = []
    for filename in os.listdir(folder):
        # print("load class")
        # print(filename, folder)
        img = cv2.imread(os.path.join(folder,filename),cv2.IMREAD_UNCHANGED)
        if img is not None:
            images.append(img)
    return images

def load_classes(folders):

    classes = {}
    for folder_class in folders:
        # print(folder_class)
        class_data = load_class_images(folder_class)
        class_name = folder_class.split('\\')[3]
        print(class_name)
        classes[class_name] = class_data
        # exit()
    return classes


def padding(img, expected_size):
    desired_size = expected_size
    delta_width = desired_size[0] - img.size[0]
    delta_height = desired_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)


def stitch(images, text):

    # images = [Image.open(x) for x in ['Test1.jpg', 'Test2.jpg', 'Test3.jpg']]
    print(type(images))
    print(images.size)
    print(images.shape)
    images_type = []
    for i in images:
        # print(i)
        print(i.size)
        print(i.shape)
        i = Image.fromarray(i)
        images_type.append(i)
    widths, heights = zip(*(i.size for i in images_type))
    print("widths: ", widths)
    print("heights: ", heights)
    total_width = sum(widths)
    max_height = max(heights)
    # lines = math.ceil(total_width / WIDTH)
    new_im = Image.new('RGB', (WIDTH, HEIGHT), color="white")
    # new_im = Image.new(mode='1', size=(total_width, max_height), color="white")

    x_offset = PADDING *  np.random.randint(1, 5, size=1)[0]
    start_offset = x_offset
    y_offset = PADDING *  np.random.randint(1, 2, size=1)[0]
    for idx, im in enumerate(images_type):
        print("x_offset: ", x_offset)
        if new_im.size[0] - (x_offset + im.size[0]) <= start_offset:
            print(int((max_height - im.size[1]) / 2))
            print(max_height)
            print(y_offset)
            y_offset = y_offset + int((max_height - im.size[1]) / 2) + max_height

            print("y_offset  ", y_offset)
            x_offset = start_offset
            print("new line: ", x_offset)
        else :
            #  CHANGE Y OFFSET BASED ON (NEW_IMG.HEIGTH - IM.HEIGTH) / 2
            # y_offset = y_offset + int((max_height - im.size[1])/2)
            print("CONSTANT: ", int((max_height - im.size[1]) / 2))
            # y_offset = y_offset + int((max_height - im.size[1])/2)
            y_offset = y_offset
        # (x_offset, 0) = upper left corner of the canvas where to paste
        print("new im size: " , new_im.size)
        print("pasted img shape: ", im.size)
        # im.show()
        #  crop(left, top, right, bottom)
        print(im.size[0])
        cropping = np.random.randint(5, 15, size=1)[0]
        if im.size[0] > 30:
            im = im.crop((cropping,  0, im.size[0]-cropping, im.size[1]))
        new_im.paste(im, (new_im.size[0] - (x_offset + im.size[0]), y_offset))
        w = im.size[0]
        h = im.size[1]
        x_c = new_im.size[0] -  (x_offset + w / 2)
        y_c = y_offset + h / 2
        box = [x_c, y_c, w, h]
        label = text[idx]
        # skip punctuations, do not save their labels, yolo should not learn them
        if label not in string.punctuation:
            save_coco_label(SCRIPT_NAME, label, box, PATH)
        print(box)
        # new_im.show()
        #  slide the upper left corner for pasting next image next iter
        x_offset = x_offset + im.size[0]+cropping
        print("x_offset end : ", x_offset, im.size[0])
    new_im.save(SCRIPT_NAME + '.png')

def resize_with_padding(img, expected_size):
    # img.thumbnail((expected_size[0], expected_size[1]))
    print(img.shape)
    delta_width = expected_size[0] - img.shape[0]
    delta_height = expected_size[1] - img.shape[1]
    pad_width = delta_width // 2
    pad_width2 = (delta_width + 1) // 2
    pad_height = delta_height // 2
    pad_height2 = (delta_height + 1) // 2
    padding = (pad_width, pad_height, delta_width + pad_width, delta_height - pad_height)
    print(pad_width, pad_height)
    # return ImageOps.expand(img, padding)
    # padding = top, bottom, left, right
    color = [255, 255, 255]
    new_im = cv2.copyMakeBorder(img, pad_height, pad_height2, pad_width , pad_width2  , cv2.BORDER_CONSTANT,
                                value=color)

    cv2.imshow("image", new_im)
    cv2.waitKey(0)
    return new_im


def resize_with_padding2(image, size):
    '''
    Resizes a black and white image to the specified size,
    adding padding to preserve the aspect ratio.
    '''
    # Get the height and width of the image
    height, width = image.shape
    print(height, width)

    # Calculate the aspect ratio of the image
    aspect_ratio = height / width

    # Calculate the new height and width after resizing to (224,224)
    new_height, new_width = size
    if aspect_ratio > 1:
        new_width = int(new_height / aspect_ratio)
    else:
        new_height = int(new_width * aspect_ratio)

    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

    # Create a black image with the target size
    padded_image = np.zeros(size, dtype=np.uint8)

    # Calculate the number of rows/columns to add as padding
    padding_rows = (size[0] - new_height) // 2
    padding_cols = (size[1] - new_width) // 2
    print(padding_rows, padding_cols)

    # Add the resized image to the padded image, with padding on the left and right sides
    padded_image[padding_rows:padding_rows + new_height, padding_cols:padding_cols + new_width] = resized_image

    return padded_image

def sample_text_generator(TEXT_LENGTH, NGRAM_SIZE):
    class_names = glob(LETTERS_FOLDER+"/*", recursive=False)
    print("class names: ", class_names)
    images = load_classes(class_names)
    print("letters keys: " , images.keys())
#     sample = images['class_name'][0]
    text = generator(TEXT_LENGTH, NGRAM_SIZE)
    # text = [elem for elem in text.split(" ") if elem != '']

    # remove all dots from text to prepare text for labelling
    # text_processed = [elem for elem in text.replace(".", "").split(" ") if elem != '']
    # print(text_processed)
    text = text.split(" ")
    script = []
    print(images)

    for letter in text:
        if letter not in string.punctuation and letter != '':
            print(images.keys())
            print(images[letter])
            random_sample_idx = np.random.choice(len(images[letter]), 1)[0]
            print("random_sample_idx :", random_sample_idx )
            random_sample = images[letter][random_sample_idx]
            print(random_sample)
            print(random_sample.shape)
            print(type(random_sample))
        else:
            end_token = np.zeros([WHITESPACE, np.random.randint(3, size=1)[0] * WHITESPACE], dtype=np.uint8)
            end_token.fill(255)
            random_sample= end_token

        script.append( random_sample)
    print("SCRIPT: ")
    print(script)
    script = np.array(script)
    return script, text

class_names = glob(LETTERS_FOLDER+"/*", recursive = False)
print(class_names)
images = load_classes(class_names)
print(images.keys())

script, text = sample_text_generator(TEXT_LENGTH, NGRAM_SIZE)
stitched = stitch(script, text)

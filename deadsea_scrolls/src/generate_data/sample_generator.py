import random
from glob import glob
import string

import numpy as np

import util.helper_functions as hf
import generate_data.augmentation.char_augmentation as ca
from generate_data.augmentation.scroll_augmentation import transform_scroll
from generate_data.ngram.text_generator_ngram import generator
from generate_data.data_generator import create_image

from util.global_params import *


def stitch(images, text, folder, script_name):
    """
    Create a new image for the script with fixed target_size. Start adding from right to left
    the selected letters from the generated text. When PADDING is reached in width (x-axis), make
    a new line by updating the y-axis.

    Parameters
    ----------
    """
    images_type = []
    for i in images:
        if i.dtype == "object":
            i = i.astype(np.uint8)
        i = hf.Image.fromarray(i)
        images_type.append(i)
    widths, heights = zip(*(i.size for i in images_type))
    max_height = max(heights)
    new_im = hf.Image.new('RGB', (WIDTH, HEIGHT), color="white")

    div_factor = random.randint(2, 4)

    x_offset = PADDING * np.random.randint(1, 4, size=1)[0]
    start_offset = x_offset
    y_offset = PADDING * np.random.randint(1, 2, size=1)[0]
    for idx, im in enumerate(images_type):
        if im.size[0] > 1 and im.size[1] > 1:
            im = im.resize((int(im.size[0] / div_factor), int(im.size[1] / div_factor)))
        if new_im.size[0] - (x_offset + im.size[0]) <= start_offset:
            y_offset = y_offset + int(max_height / div_factor)
            x_offset = start_offset
        else:
            #  CHANGE Y OFFSET BASED ON (NEW_IMG.HEIGTH - IM.HEIGTH) / 2
            y_offset = y_offset
            # (x_offset, 0) = upper left corner of the canvas where to paste
        # cropping = np.random.randint(5, 8, target_size=1)[0]
        if y_offset + im.size[1] >= HEIGHT:
            break
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
        label = text[idx]
        # skip punctuations, do not save their labels, yolo should not learn them
        if label not in string.punctuation:
            hf.save_coco_label(script_name, label, box, DATA_FOLDER, folder)
            # uncomment to draw boxes
            # hf.draw_boxes(new_im, x_c, y_c, w, h, idx)
        # uncomment to see the process of stitching
        # new_im.show()
        #  slide the upper left corner for pasting next image next iter
        x_offset = x_offset + im.size[0]

    # augment the final image
    new_im = transform_scroll(new_im)

    folder_path = os.path.join(DATA_FOLDER, 'images', folder)

    new_im.save(
        os.path.join(folder_path, script_name + '.png'))


def sample_text_generator(text_len, ngram_size):
    """
    Main function calling all the other helper function.
    Generate text, transform it into script.

    Parameters
    ----------
    """
    class_names = glob(LETTERS_FOLDER + os.sep + "*", recursive=False)
    images = hf.load_classes(class_names)
    text = generator(text_len, ngram_size)

    # remove all dots from text to prepare text for labelling
    text = text.split(" ")
    script = []

    # set the parameters for the image augmentation
    params = ca.get_random_param_values()

    for letter in text:
        if letter not in string.punctuation and letter != '':
            random_sample_idx = np.random.choice(len(images[letter]), 1)[0]

            if np.random.random_sample() < 0.5:
                random_sample = create_image(letter, (64, 69)).convert('1')
                random_sample = np.array(random_sample, dtype=np.uint8)
            else:
                # TODO: these are grayscale images (uint8 ndarray)
                #  make them binary
                random_sample = images[letter][random_sample_idx]
            # add some transformations to the image (letter)
            random_sample = ca.transform_letter(random_sample, *params)
        else:
            end_token = np.zeros(
                [WHITESPACE, np.random.randint(1, 3, size=1)[0] * WHITESPACE],
                dtype=np.uint8)
            end_token.fill(255)
            random_sample = end_token

        script.append(random_sample)

    script = np.asarray(script, dtype=object)
    return script, text


def generate_sample(folder, script_name, text_length=TEXT_LENGTH):
    script, text = sample_text_generator(text_length, NGRAM_SIZE)

    # THIS IS VYVY'S PART
    # for im in script:
    #     print(im.shape)
    # rotate([im for im in script if im.shape[0] > 20])

    stitch(script, text, folder, script_name)

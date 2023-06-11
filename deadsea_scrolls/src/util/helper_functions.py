"""
Some utility functions for generating data.
"""
import os
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, List

Box = List[float]
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
    Normalizes images target_size to WIDTH and HEIGHT

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


def save_coco_label(file: str, label_class: str, points: Box, data_folder: str,
                    folder: str):
    """
    Saves the true_label of the image in coco format: classs, x_c, y_c, w, h

    Parameters
    ----------
        file (str): Name of true_label (same as name of image)
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
    with open(os.path.join(data_folder, "labels", folder, str(file)), 'a') as f:
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


def assert_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def remove_directory(directory):
    # check if the directory exists
    if not os.path.isdir(directory):
        return

    # Iterate over the contents of the directory
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        print(item_path)

        # Check if the item is a directory
        if os.path.isdir(item_path):
            # Recursively call remove_directory to remove subdirectories
            remove_directory(item_path)

        else:
            # Remove files within the directory
            os.remove(item_path)
            # print(f"Removed file: {item_path}")

    # Remove the empty directory itself
    os.rmdir(directory)
    # print(f"Removed directory: {directory}")


# TODO: might use this function in the final pipeline
# def padding(img, expected_size):
#     """
#     Make a dict with {key: 'letter_name', value: list[imgs]}
#
#     Parameters
#     ----------
#     """
#     desired_size = expected_size
#     delta_width = desired_size[0] - img.target_size[0]
#     delta_height = desired_size[1] - img.target_size[1]
#     pad_width = delta_width // 2
#     pad_height = delta_height // 2
#     padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
#     return ImageOps.expand(img, padding)


def draw_boxes(img, x, y, w, h, index):
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
    assert_dir("boxes")
    path = os.path.join("boxes", f"image{index}.png")
    # save the image
    cv2.imwrite(path, image)
    # cv2.imshow('image', image)
    # Displaying the image
    # cv2.waitKey(0)

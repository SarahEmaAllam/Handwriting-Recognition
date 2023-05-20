import random

import imageio
import numpy as np
import imgaug as ia
import imageio.v3 as iio
import imgaug.augmenters as iaa
import cv2
import albumentations as A
from wezel.main import assert_dir
from PIL import Image
import os

ia.seed(1)


def gen_random_img_list(path, list_len):
    random_dir = random.choice(os.listdir(path))
    random_file = random.choice(os.listdir(os.path.join(path, random_dir)))
    img_list = []
    for i in range(0, list_len):
        print(i)
        img = iio.imread(os.path.join(path, random_dir, random_file))
        img_list.append(img)
    return img_list


def read_images(path):
    path = '../data/preprocessed_images/symbols'
    dir = os.listdir(path)
    file = os.listdir(os.path.join(path, dir))

    # read and show image
    img = iio.imread(os.path.join(path, dir, file))
    img_list = [img, img, img, img, img]
    files = os.listdir(os.path.join(path))

    img = iio.imread(os.path.join(path, random_file))
    img_list = [img, img, img, img, img]


def rotate_by_degree(img_list):
    nr = len(img_list)
    print(nr)
    degree_diff = 90 / (nr - 1)
    degree_start = -degree_diff
    degree_end = 0
    aug_img_list = []
    for img in img_list:
        rotate = iaa.Affine(rotate=(degree_start, degree_end), mode='constant')
        image_aug = rotate(image=img)
        ia.imshow(image_aug)
        print('start', degree_start)
        print('end', degree_end)
        degree_start = degree_end
        degree_end += degree_diff
        aug_img_list.append(image_aug)

    ia.imshow(np.hstack(aug_img_list))


def rotate_several_by_degree(img_list):
    nr_lines = random.choice(4, 1)
    for line in nr_lines:
        pass

def random_rotation_by_list(path):
    # choose a random directory and file
    random_file = random.choice(os.listdir(os.path.join(path)))

    # read and show image
    img = iio.imread(os.path.join(path, random_file))
    img_list = []
    for i in range(0, 4):
        random_file = random.choice(os.listdir(os.path.join(path)))
        img_list.append(random_file)
    ia.seed(4)
    aug_img_list = []
    rotate = iaa.Affine(rotate=(-135, 135), mode='constant')

    img_list = [img, img, img, img, img]
    image_aug = rotate(images=img_list)
    ia.imshow(np.hstack(image_aug))


if __name__ == '__main__':
    path = '../../Ressources/'
    img_list = gen_random_img_list(path, 5)
    rotate_by_degree(img_list)

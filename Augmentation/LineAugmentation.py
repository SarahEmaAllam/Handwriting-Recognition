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


def read_images(path):
    # srcimage = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    image = cv2.cvtColor(cv2.imread(path, -1), cv2.COLOR_BGR2RGB)
    # Declare an augmentation pipeline
    transform = A.Compose([
        A.RandomCrop(width=35, height=42),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
    ])
    transformed = transform(image=image)
    transformed_image = transformed["image"]


def rotate_by_degree(img_list):
    nr = len(img_list)
    print(nr)
    degree_diff = 90 / nr
    degree_start = -degree_diff
    degree_end=0
    aug_img_list = []
    for img in img_list:
        rotate = iaa.Affine(rotate=(degree_start, degree_end), mode='constant')
        image_aug = rotate(image=img)
        ia.imshow(image_aug)
        aug_img_list.append(image_aug)
        print('start', degree_start)
        print('end', degree_end)
        degree_start = degree_end
        degree_end += degree_diff
        aug_img_list.append(image_aug)

    ia.imshow(np.hstack(aug_img_list))


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
    # path = 'navis-QIrug-Qumran_extr09_0001-line-008-y1=400-y2=515-zone-HUMAN-x=1650-y=0049-w=0035-h=0042-ybas=0027-nink=631-segm=COCOS5cocos.pgm'
    # srcimage = cv2.imread(path, -1)
    # srcimage = srcimage.astype(np.uint8)

    path = 'ressources'
    random_file = random.choice(os.listdir(os.path.join(path)))

    img = iio.imread(os.path.join(path, random_file))
    img_list = [img, img, img, img, img]
    rotate_by_degree(img_list)
    # ia.imshow(image_aug)

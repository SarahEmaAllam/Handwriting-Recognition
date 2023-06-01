import random
import imageio
import numpy as np
import imgaug as ia
import imageio.v3 as iio
import imgaug.augmenters as iaa
import cv2
# import albumentations as A
# from generate_data.helper_functions import assert_dir
from PIL import Image
import os

ia.seed(1)


def gen_random_img_list(path, list_len):
    img_list = []
    for i in range(0, list_len):
        random_dir = random.choice(os.listdir(path))
        random_file = random.choice(os.listdir(os.path.join(path, random_dir)))
        img = iio.imread(os.path.join(path, random_dir, random_file))
        img_list.append(img)
    return img_list


def rotate_by_degree(img_list):
    nr = len(img_list)
    print('listlänge',nr)
    degree_diff = 90 / (nr - 1) if nr > 1 else nr
    degree_start = -degree_diff
    degree_end = 0
    aug_img_list = []
    for img in img_list:
        rotate = iaa.Affine(rotate=(degree_start, degree_end), mode='constant')
        image_aug = rotate(image=img)
        print('start', degree_start)
        print('end', degree_end)
        degree_start = degree_end
        degree_end += degree_diff
        aug_img_list.append(image_aug)
    # ia.imshow(np.hstack(aug_img_list))
    return aug_img_list


def rotate_several_by_degree(img_list):
    def chunks(l, n):
        """Yield n number of striped chunks from l."""
        for i in range(0, n):
            yield l[i::n]

    nr_lines = random.choice((4, 1))
    final_img_list = []
    for line in range(nr_lines):
        rand_chunks = chunks(img_list, nr_lines)
        for chunk in rand_chunks:
            #ia.imshow(np.hstack(chunk))
            aug_rand_sample = rotate_by_degree(chunk)
            #ia.imshow(np.hstack(aug_rand_sample))
            final_img_list += aug_rand_sample
    return final_img_list


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

def rotate(img_list):
    ia.imshow(np.hstack(rotate_several_by_degree(img_list)))

if __name__ == '__main__':
    path = '../../Ressources/'
    img_list = gen_random_img_list(path, 35)
    #rotate_several_by_degree(img_list)
    ia.imshow(np.hstack(rotate_several_by_degree(img_list)))
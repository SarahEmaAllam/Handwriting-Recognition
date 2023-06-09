import os
import time

import cv2
import numpy as np


from util.utils import set_working_dir
from util.global_params import IMAGES_PATH, BINARIZED_IMAGES_PATH

DATA_PATH = "data/"


def assert_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def is_image_file(filename):
    return filename.lower().endswith(
        ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif', '.pgm'))


def binarize_images(inputdir, outputdir):
    print("inputdir : ", inputdir)
    print("outputdir ", outputdir)
    print("curent  : ", os.getcwd())
    print(*os.walk(inputdir))
    for val in os.walk(inputdir):
        print(val)
    for subdir, _, files in os.walk(inputdir):
        print("Start binarize: " + os.path.basename(subdir))
        fulloutpath = os.path.join(outputdir, os.path.basename(subdir))

        if len(files) != 0:
            assert_dir(fulloutpath)

        for file in files:
            if not is_image_file(file):
                print(type(file))
                continue

            filepath = os.path.join(subdir, file)
            srcimage = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            thresholded = cv2.threshold(srcimage, 254, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_BINARY)[1]
            amount, labels, stats, _ = cv2.connectedComponentsWithStats(thresholded, 4, cv2.CV_32S)

            dimensions = srcimage.shape
            newimg = np.full(dimensions, 255, np.uint8)
            for label in range(1, amount):
                component = stats[label]
                left = component[cv2.CC_STAT_LEFT]
                up = component[cv2.CC_STAT_TOP]
                width = component[cv2.CC_STAT_WIDTH]
                right = left + width
                height = component[cv2.CC_STAT_HEIGHT]
                down = up + height
                component_img = srcimage[up:down, left:right]
                thresholded_comp = cv2.threshold(component_img, 254, 255, cv2.THRESH_OTSU)[1]
                newimg[up:down, left:right] = thresholded_comp

            cv2.imwrite(os.path.join(fulloutpath, file), newimg)

        print("Finished binarize: " + os.path.basename(subdir))


def binarize(show_progress=False):

    assert_dir(DATA_PATH)
    assert_dir(BINARIZED_IMAGES_PATH)

    time_start = time.time()
    if show_progress:
        print("Starting binarize")

    binarize_images(IMAGES_PATH, BINARIZED_IMAGES_PATH)

    if show_progress:
        print("Finished binarize in " + str(time.time() - time_start) + " seconds")


if __name__ == '__main__':
    # change working dir to root of project
    set_working_dir(os.path.abspath(__file__))

    binarize(show_progress=True)

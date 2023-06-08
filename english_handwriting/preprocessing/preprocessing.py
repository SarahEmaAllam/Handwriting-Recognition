import os
import cv2
import numpy as np

PREPROCESS_DIR = "../data/"
SOURCE_DATA = "../data/IAM-data/"


def assert_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def is_image_file(filename):
    return filename.lower().endswith(
        ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif', '.pgm'))


def binarize_images(inputdir, outputdir):

    for subdir, _, files in os.walk(inputdir):
        print("Start binarize: " + os.path.basename(subdir))
        fulloutpath = os.path.join(outputdir, os.path.basename(subdir))

        if len(files) != 0:
            assert_dir(fulloutpath)

        for file in files:
            if not is_image_file(file):
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


def preprocessing():

    assert_dir(PREPROCESS_DIR)

    binarized_dir = os.path.join(PREPROCESS_DIR, "binarized")
    assert_dir(binarized_dir)

    print("Starting binarize")
    binarize_images(SOURCE_DATA, binarized_dir)
    print("Finished binarize")


if __name__ == "__main__":
    preprocessing()

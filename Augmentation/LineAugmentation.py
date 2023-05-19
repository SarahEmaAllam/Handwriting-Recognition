import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import cv2
from wezel.main import assert_dir

ia.seed(1)


def read_images(path):
    #srcimage = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    cv2.imread(path,-1)

seq = iaa.Sequential([

])

import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import cv2
import albumentations as A
from wezel.main import assert_dir

ia.seed(1)


def read_images(path):
    # srcimage = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    image = cv2.cvtColor(cv2.imread(path, -1), cv2.COLOR_BGR2RGB)
    # Declare an augmentation pipeline
    transform = A.Compose([
        A.RandomCrop(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
    ])
    transformed = transform(image=image)
    transformed_image = transformed["image"]

if __name__ == '__main__':
    read_images('navis-QIrug-Qumran_extr09_0001-line-008-y1=400-y2=515-zone-HUMAN-x=1650-y=0049-w=0035-h=0042-ybas=0027-nink=631-segm=COCOS5cocos.pgm')

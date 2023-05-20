import imageio
import numpy as np
import imgaug as ia
import imageio.v3 as iio
import imgaug.augmenters as iaa
import cv2
import albumentations as A
from wezel.main import assert_dir
from PIL import Image

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


if __name__ == '__main__':
    '''aug_img = read_images(
        'navis-QIrug-Qumran_extr09_0001-line-008-y1=400-y2=515-zone-HUMAN-x=1650-y=0049-w=0035-h=0042-ybas=0027-nink=631-segm=COCOS5cocos.pgm')
    #aug_img = aug_img.astype(np.uint8)
    cv2.imshow('augmented', aug_img)
    # show all images
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    # path = 'navis-QIrug-Qumran_extr09_0001-line-008-y1=400-y2=515-zone-HUMAN-x=1650-y=0049-w=0035-h=0042-ybas=0027-nink=631-segm=COCOS5cocos.pgm'
    # srcimage = cv2.imread(path, -1)
    # srcimage = srcimage.astype(np.uint8)
    path = 'pics/navis-QIrug-Qumran_extr09_0001-line-008-y1=400-y2=515-zone-HUMAN-x=1650-y=0049-w=0035-h=0042-ybas=0027-nink=631-segm=COCOS5cocos.jpg'
    im = iio.imread(path)
    ia.imshow(im)

    ia.seed(4)

    rotate = iaa.Affine(rotate=(-25, 25))
    image_aug = rotate(image=im)

    print("Augmented:")
    ia.imshow(image_aug)

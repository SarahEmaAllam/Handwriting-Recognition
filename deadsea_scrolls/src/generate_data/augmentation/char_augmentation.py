"""
Some functions for character-level augmentation.
"""
import random
import imgaug.augmenters as iaa
import numpy as np
from PIL import Image
import cv2


def get_random_param_values():
    """
    Get random values for the parameters of the image augmentation.
    The same values are used for all the letters in a script.
    """
    # random values for the parameters of the image augmentation
    rotation = (-15, 15)
    shear = (-15, 15)
    gauss_blur_sigma = (0, 2.0)
    crop = (-0.2, 0.2)
    elastic_alpha = (0, 20)
    elastic_sigma = (4, 6)

    # for each parameter, select a random value from the range
    rotation = random.uniform(*rotation)
    shear = random.uniform(*shear)
    gauss_blur_sigma = random.uniform(*gauss_blur_sigma)
    crop = random.uniform(*crop)
    elastic_alpha = random.uniform(*elastic_alpha)
    elastic_sigma = random.uniform(*elastic_sigma)

    return rotation, shear, gauss_blur_sigma, crop, elastic_alpha, elastic_sigma


def pil_to_ndarray(image) -> np.ndarray:
    """
    Convert a PIL image to a numpy array.
    """
    image = np.asarray(image)
    return image


def ndarray_to_pil(image: np.ndarray) -> Image:
    """
    Convert a numpy array to a PIL image.
    """
    image = Image.fromarray(image)
    return image


def to_uint8(image: np.ndarray) -> np.ndarray:
    """
    Convert an image to uint8.
    """
    if image.dtype != np.uint8:
        image = image.astype(np.uint8) * 255

    return image


def add_diagonal_cuts(images, random_state, parents, hooks):
    """Add diagonal white cuts through the letter"""
    augm_imgs = []

    for image in images:
        number_of_cuts = random_state.randint(0, 4)

        for _ in range(number_of_cuts):
            # Choose a random point on the left margin of the image
            y_left = random_state.randint(0, image.shape[1])

            mid_y = image.shape[1] // 2

            # Choose a random point on the right margin of the image
            # The point must be on the opposite side of the image to
            #  simulate a diagonal cut
            if y_left < mid_y:
                y_right = random_state.randint(mid_y, image.shape[1])
            else:
                y_right = random_state.randint(0, mid_y)

            # Define the points to draw the line
            point_left = (0, y_left)
            point_right = (image.shape[0], y_right)

            # Choose a random thickness for the line
            thickness = random_state.randint(1, 4)

            # Draw the line on the mask
            image = cv2.line(
                image, point_left, point_right, (255, 255, 255), thickness)

        augm_imgs.append(image)

    return augm_imgs


def transform_letter(image: np.ndarray, rotation: float, shear: float,
                     gauss_blur_sigma: float, crop: float, elastic_alpha: float,
                     elastic_sigma: float) -> np.ndarray:
    """
    Apply a series of transformations to the letter image
    and return the augmented image.
    """
    image = pil_to_ndarray(image)
    image = to_uint8(image)

    # the same augmentation applied to be applied to each letter image in a script
    aug = iaa.Sequential([
        iaa.Affine(rotate=rotation, shear=shear, mode='constant', cval=255),
        iaa.GaussianBlur(sigma=gauss_blur_sigma),
        iaa.KeepSizeByResize(
            iaa.CropAndPad(percent=crop, pad_mode='constant', pad_cval=255)),
        iaa.Pad(px=2, pad_mode='constant', pad_cval=255),
        iaa.ElasticTransformation(alpha=elastic_alpha, sigma=elastic_sigma),
        iaa.size.Crop(px=2),
        iaa.Sometimes(
            0.5,
            iaa.Lambda(add_diagonal_cuts)
        ),
        iaa.Sometimes(
            0.1,
            iaa.KeepSizeByResize(
                iaa.CropAndPad(percent=(-0.5, 0.5), pad_mode='constant',
                               pad_cval=255))
        )
    ])

    aug_image = aug(image=image)

    return aug_image



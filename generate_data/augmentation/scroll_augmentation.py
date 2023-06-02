from generate_data.augmentation.char_augmentation import *


def transform_scroll(image) -> Image:
    """
    Apply a series of transformations to the script image and return the augmented image.
    """
    image = pil_to_ndarray(image)

    # random cutouts with white pixels to simulate noise/scroll erosion
    cutout = iaa.Cutout(nb_iterations=(50, 100), size=(0.01, 0.05),
                        squared=False, fill_mode="constant", cval=255)

    aug_image = cutout(image=image)
    aug_image = ndarray_to_pil(aug_image)

    return aug_image

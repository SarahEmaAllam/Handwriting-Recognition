import random
import cv2
import numpy as np
import albumentations as A


# gets PIL image and returns augmented PIL image
def augment_img(img: np.ndarray):

    # only augment 3/4th the images
    if random.randint(1, 4) > 3:
        return img

    # morphological alterations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    if random.randint(1, 5) == 1:
        # dilation because the image is not inverted
        img = cv2.erode(img, kernel, iterations=random.randint(1, 2))
    if random.randint(1, 6) == 1:
        # erosion because the image is not inverted
        img = cv2.dilate(img, kernel, iterations=random.randint(1, 1))

    # img = Image.fromarray(img)

    # noise introduction
    transform = A.Compose([

        A.OneOf([
            # add black pixels noise
            A.OneOf([
                # A.RandomRain(brightness_coefficient=1.0, drop_length=2, drop_width=2, drop_color = (0, 0, 0), blur_value=1, rain_type = 'drizzle', p=0.05),
                #  A.RandomShadow(p=1),
                A.PixelDropout(p=1),
            ], p=0.9),

            # add white pixels noise
            # A.OneOf([
            #     A.RandomRain(brightness_coefficient=1.0, drop_length=2,
            #                  drop_width=2, drop_color=(255, 255, 255),
            #                  blur_value=1, rain_type=None, p=1),
            # ], p=0.9),
        ], p=1),

        # transformations
        A.OneOf([
            A.ShiftScaleRotate(shift_limit=0, scale_limit=0.25, rotate_limit=2,
                               border_mode=cv2.BORDER_CONSTANT,
                               value=(255, 255, 255), p=1),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0, rotate_limit=8,
                               border_mode=cv2.BORDER_CONSTANT,
                               value=(255, 255, 255), p=1),
            A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.15,
                               rotate_limit=11, border_mode=cv2.BORDER_CONSTANT,
                               value=(255, 255, 255), p=1),
            A.Affine(shear=random.randint(-5, 5), mode=cv2.BORDER_CONSTANT,
                     cval=(255, 255, 255), p=1),
            A.ElasticTransform(alpha=random.randint(0, 10),
                               sigma=random.randint(0, 100)),
        ], p=0.5),
        # A.Blur(blur_limit=5,p=0.25),
    ])
    img = transform(image=img)

    return img


# def augment_using_ops(images, labels):
# 	# randomly flip the images horizontally, randomly flip the images
# 	# vertically, and rotate the images by 90 degrees in the counter
# 	# clockwise direction
# 	images = tf.image.random_flip_left_right(images)
# 	images = tf.image.random_flip_up_down(images)
# 	images = tf.image.rot90(images)
# 	# return the image and the label
# 	return (images, labels)

# def load_batch():
#     ds = tf.data.Dataset.from_tensor_slices(imagePaths)
#     ds = (ds
#           # .shuffle(len(imagePaths), seed=42)
#           # .map(load_images, num_parallel_calls=AUTOTUNE)
#           # .cache()
#           # .batch(BATCH_SIZE)
#           # call the augmentation method here
#           .map(augment_img, num_parallel_calls=AUTOTUNE)
#           # .prefetch(tf.data.AUTOTUNE)
#           )
#
#     return ds




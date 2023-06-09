import random
import cv2
import numpy as np
import albumentations as A
from preprocessing import preprocessing
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from model import Model

# gets PIL image and returns augmented PIL image
def augment_img(img):
    # only augment 3/4th the images
    if random.randint(1, 4) > 3:
        return img

    img = np.asarray(img)  # convert to numpy for opencv

    # morphological alterations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    if random.randint(1, 5) == 1:
        # dilation because the image is not inverted
        img = cv2.erode(img, kernel, iterations=random.randint(1, 2))
    if random.randint(1, 6) == 1:
        # erosion because the image is not inverted
        img = cv2.dilate(img, kernel, iterations=random.randint(1, 1))

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
            A.OneOf([
                A.RandomRain(brightness_coefficient=1.0, drop_length=2,
                             drop_width=2, drop_color=(255, 255, 255),
                             blur_value=1, rain_type=None, p=1),
            ], p=0.9),
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
    # img = transform(image=img)['image']
    # image = Image.fromarray(img)
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

def test():
    print("[INFO] initializing model...")
    # model = Sequential()
    # model.add(keras.layers.Conv2D(32, (3, 3), padding="same", input_shape=(32, 32, 3)))
    # model.add(keras.layers.Activation("relu"))
    # model.add(keras.layers.Flatten())
    # model.add(keras.layers.Dense(10))
    # model.add(keras.layers.Activation("softmax"))

    print("[INFO] compiling model...")
    # model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd",
    #               metrics=["accuracy"])

    model = Model().build_model()
    # train the model
    print("[INFO] training model...")

    train_batches, val_batches, test_batches, decoder = preprocessing.preprocess()
    H = model.fit(
        train_batches.take(1),
        validation_data=val_batches.take(1),
        epochs=10)

    print(H)
# testDs =
# # show the accuracy on the testing set
# (loss, accuracy) = model.evaluate(testDS)
# print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))

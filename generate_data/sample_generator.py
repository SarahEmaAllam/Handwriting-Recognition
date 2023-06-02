from glob import glob
from generate_data.ngram.text_generator_ngram import generator
import string
from generate_data.data_generator import create_image
from generate_data.helper_functions import *
from generate_data.augmentation.scroll_augmentation import *
from generate_data.augmentation.line_augmentation import *

# should be based on N-gram probability distribution
FOLDER = 'new'
WORD_LENGTH = 10
TEXT_LENGTH = 100 * np.random.randint(1, 5, size=1)[0]
# text_len = 10
# SCRIPT_SIZE = 608
NGRAM_SIZE = 4
Box = [float, float, float, float]
WIDTH = 640
HEIGHT = 640
PADDING = 10 * np.random.randint(10, size=1)[0]
WHITESPACE = 15
PATH = os.getcwd()
SCRIPT_NAME = 'test2'
# LETTERS_FOLDER = os.path.join('preprocess', 'output', 'symbols')

DATA_FOLDER = 'data'
LETTERS_FOLDER = 'data/preprocessed_images/symbols'


def stitch(images, text, folder, script_name):
    """
    Create a new image for the script with fixed size. Start adding from right to left
    the selected letters from the generated text. When PADDING is reached in width (x-axis), make
    a new line by updating the y-axis.

    Parameters
    ----------
    """
    images_type = []
    for i in images:
        i = Image.fromarray(i)
        images_type.append(i)
    widths, heights = zip(*(i.size for i in images_type))
    max_height = max(heights)
    new_im = Image.new('RGB', (WIDTH, HEIGHT), color="white")

    x_offset = PADDING * np.random.randint(1, 4, size=1)[0]
    start_offset = x_offset
    y_offset = PADDING * np.random.randint(1, 2, size=1)[0]
    for idx, im in enumerate(images_type):
        if im.size[0] > 1 and im.size[1] > 1:
            im = im.resize((int(im.size[0] / 2), int(im.size[1] / 2)))
        if new_im.size[0] - (x_offset + im.size[0]) <= start_offset:
            y_offset = y_offset + int(max_height / 2)
            x_offset = start_offset
        else:
            #  CHANGE Y OFFSET BASED ON (NEW_IMG.HEIGTH - IM.HEIGTH) / 2
            y_offset = y_offset
            # (x_offset, 0) = upper left corner of the canvas where to paste
        # cropping = np.random.randint(5, 8, size=1)[0]
        if y_offset >= HEIGHT:
            break
        cropping = np.random.randint(5, 6, size=1)[0]
        if im.size[0] > 10:
            im = im.crop((cropping, 0, im.size[0] - cropping, im.size[1]))
        new_im.paste(im, (new_im.size[0] - (x_offset + im.size[0]), y_offset))
        w = im.size[0]
        h = im.size[1]
        x_c = (new_im.size[0] - (x_offset + w / 2)) / WIDTH
        y_c = (y_offset + h / 2) / HEIGHT
        w = w / WIDTH
        h = h / HEIGHT
        box = [x_c, y_c, w, h]
        label = text[idx]
        assert_dir(DATA_FOLDER)
        # skip punctuations, do not save their labels, yolo should not learn them
        if label not in string.punctuation:
            save_coco_label(script_name, label, box, DATA_FOLDER, folder)
            # uncomment to draw boxes
            # draw_boxes(new_im, x_c, y_c, w, h, idx)
        # uncomment to see the process of stitching
        # new_im.show()
        #  slide the upper left corner for pasting next image next iter
        x_offset = x_offset + im.size[0]

    # augment the final image
    new_im = transform_scroll(new_im)

    folder_path = os.path.join(DATA_FOLDER, 'images', folder)
    assert_dir(folder_path)

    new_im.save(
        os.path.join(folder_path, script_name + '.png'))


def sample_text_generator(text_len, ngram_size):
    """
    Main function calling all the other helper function.
    Generate text, transform it into script.

    Parameters
    ----------
    """
    class_names = glob(LETTERS_FOLDER + os.sep + "*", recursive=False)
    images = load_classes(class_names)
    text = generator(text_len, ngram_size)

    # remove all dots from text to prepare text for labelling
    text = text.split(" ")
    script = []

    # set the parameters for the image augmentation
    params = get_random_param_values()

    for letter in text:
        if letter not in string.punctuation and letter != '':
            random_sample_idx = np.random.choice(len(images[letter]), 1)[0]

            if np.random.random_sample() < 0.5:
                random_sample = create_image(letter, (64, 69)).convert('1')
                random_sample = np.array(random_sample)
            else:
                # TODO: these are grayscale images (uint8 ndarray)
                #  make them binary
                random_sample = images[letter][random_sample_idx]
            # add some transformations to the image (letter)
            random_sample = transform_letter(random_sample, *params)
        else:
            end_token = np.zeros(
                [WHITESPACE, np.random.randint(1, 3, size=1)[0] * WHITESPACE],
                dtype=np.uint8)
            end_token.fill(255)
            random_sample = end_token

        script.append(random_sample)

    script = np.array(script, dtype=object)
    return script, text


def generate_sample(folder, script_name, text_length=TEXT_LENGTH):
    # print("\nGenerating sample: ", script_name)
    class_names = glob(LETTERS_FOLDER + os.sep + "*", recursive=False)
    # print(class_names)

    # images = load_classes(class_names)
    script, text = sample_text_generator(text_length, NGRAM_SIZE)

    # THIS IS VYVY'S PART
    # for im in script:
    #     print(im.shape)
    # rotate([im for im in script if im.shape[0] > 20])

    stitch(script, text, folder, script_name)


# for i in range(10):
#     generate_sample(FOLDER, f"sample{i}")

# generate_sample(FOLDER, "test")




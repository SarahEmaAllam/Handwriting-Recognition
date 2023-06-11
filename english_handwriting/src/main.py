import os

import sys
import argparse
from preprocessing.preprocessing import preprocess_test_images
from testing import testing


def parse():
    """
    Reads input either from console or as a python script
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--p", "--path", action="store_true",
                        help="Path to line image")
    # args = parser.parse_args()


def get_filenames(path):
    """
    Extract filenames from given directory
    :param path: path to directory
    :return: list of filenames
    """
    all_files = os.listdir(path)
    filenames = []
    for file in all_files:
        f_name = os.path.basename(file)
        filenames.append(os.path.splitext(f_name)[0])
    return filenames


def text_to_file(text, path):
    """
    Generates a txt file for each predicted line
    Saves file in 'results' directory
    Creates 'results' directory in parental directory if non-existent
    :param text: list of predicted lines
    :param filenames: list of filenames
    """
    # format: ./results/img_001_characters.txt
    save_path = './results'
    filenames = get_filenames(path)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for idx, line in enumerate(text):
        name = filenames[idx]
        complete_path = os.path.join(save_path, name + '.txt')
        with open(complete_path, 'w') as f:
            f.write(f"{line}\n")


if __name__ == '__main__':
    # reads input either from console or as a python script
    parse()
    if len(sys.argv) > 1:
        try:
            path = sys.argv[1]
        except IndexError:
            exit("Path couldn't be resolved.")
    else:
        path = input("Path to line image:")

    images_names, test_images, decoder = preprocess_test_images(path)
    outputs = testing(test_images, decoder)
    text_to_file(outputs, path)


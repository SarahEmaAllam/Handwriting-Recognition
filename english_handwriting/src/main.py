import os
from preprocessing.binarization import binarize
from preprocessing.preprocessing import preprocess
from util.utils import set_working_dir

if __name__ == '__main__':
    # change working dir to root of project
    set_working_dir(os.path.abspath(__file__))

    # TODO: check if the binerized images are already created
    # binarize()
    preprocess()

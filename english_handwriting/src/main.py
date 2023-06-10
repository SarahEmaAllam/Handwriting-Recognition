import os
from preprocessing.binarization import binarize
from preprocessing.preprocessing import preprocess
from util.utils import set_working_dir
from training import train

if __name__ == '__main__':
    # change working dir to root of project
    set_working_dir(os.path.abspath(__file__))
    # os.chdir(os.path.dirname (os.path.abspath(__file__)))
    train()

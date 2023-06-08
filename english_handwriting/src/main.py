import os
# import model ...
from preprocessing.binarization import binarize
from preprocessing.preprocessing import preprocess_data

if __name__ == '__main__':
    # change working dir to root of project
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.chdir('../')

    # TODO: check if the binerized images are already created
    # binarize()
    preprocess_data(print_progress=True)

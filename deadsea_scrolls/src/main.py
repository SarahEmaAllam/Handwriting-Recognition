from yolo.model import run_model
import os


if __name__ == '__main__':

    # change working dir to root of project
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.chdir('../')

    run_model()
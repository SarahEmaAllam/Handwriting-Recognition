from yolo.model import run_model
import os
import sys


if __name__ == '__main__':

    # change working dir to root of project
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.chdir('../')

    try:
        arg_command = sys.argv[1]
    except IndexError:
        arg_command = "all"

    run_model(arg_command)

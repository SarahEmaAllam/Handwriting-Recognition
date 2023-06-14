from yolo.model import run_model
from yolo.predict import predict
from ultralytics.yolo.utils import set_settings
from util.global_params import *
import os
import sys

if __name__ == '__main__':

    # change working dir to root of project
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.chdir('../')

    # check if there is an input given to the program
    try:
        arg_command = sys.argv[1]
    except IndexError:
        exit("This folder does not exist.\n"
             "Use | generate | train | all | to train the model.\n"
             "Or give a | path | relative path | to do a prediction on a set of images.")


    # fix to force yolo to use the directories we want it to use
    set_settings({'datasets_dir': os.path.join(os.getcwd(), DATA_FOLDER),
                  'runs_dir': os.path.join(os.getcwd(), RUN_FOLDER)})

    # if the input is a command for the model run it
    if arg_command in ["generate", "train", "all"]:
        run_model(arg_command)
        exit()

    # check if input is a directory
    file_path = arg_command.strip('\"')
    if not os.path.isdir(file_path):
        if not os.path.isdir(os.path.join(".", file_path)):
            exit("This folder does not exist.\n"
                 "Use | generate | train | all | to train the model.\n"
                 "Or give a | path | relative path | to do a prediction on a set of images.")

    predict(arg_command)

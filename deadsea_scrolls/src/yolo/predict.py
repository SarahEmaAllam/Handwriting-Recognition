import gdown
import validators

from util.global_params import *
import os
import numpy as np
from ultralytics import YOLO
import util.helper_functions as hf

reverse_labels = {
    0: 'א',
    1: 'ע',
    2: 'ב',
    3: 'ד',
    4: 'ג',
    5: 'ה',
    6: 'ח',
    7: 'כ',
    8: 'ך',
    9: 'ל',
    10: 'ם',
    11: 'מ',
    12: 'ן',
    13: 'נ',
    14: 'פ',
    15: 'ף',
    16: 'ק',
    17: 'ר',
    18: 'ס',
    19: 'ש',
    20: 'ת',
    21: 'ט',
    22: 'ץ',
    23: 'צ',
    24: 'ו',
    25: 'י',
    26: 'ז'}


def predict(file_folder):

    # set up the model

    prediction_model = PREDICTION_MODEL
    if validators.url(PREDICTION_MODEL):
        prediction_model = 'resources/yolo_downloaded'
        gdown.download(PREDICTION_MODEL, prediction_model, quiet=False)
    model = YOLO(prediction_model)

    # check if results directory exists
    hf.assert_dir("results")

    # delete and recreate current if it already exists
    prediction_dir = os.path.join(PREDICTION_DIR, "current")
    if os.path.exists(prediction_dir):
        hf.remove_directory(prediction_dir)

    # create current dir
    os.makedirs(prediction_dir)

    #
    yolo_predict_dir = os.path.join(RUN_FOLDER, "detect", "predict")
    if os.path.exists(yolo_predict_dir):
        hf.remove_directory(yolo_predict_dir)

    # predict all binarized images and print output to a txt file
    for file in os.listdir(file_folder):

        if "binarized" in file:

            # do the prediction
            file_path = os.path.join(file_folder, file)
            boxes = model.predict(file_path, save=True)[0].boxes.boxes.cpu().detach().numpy()

            # Calculate maximum bounding box height
            max_height = np.max(boxes[::, 3] - boxes[::, 1]) / 2

            # Sort the bounding boxes by y-value
            by_y = sorted(boxes, key=lambda y: y[1])  # y values

            line_y = by_y[0][1]  # first y

            # Assign a line number to each bounding box
            by_line = []
            line = 1
            for x, y, w, h, _, a in by_y:
                if y > line_y + max_height:
                    line_y = y
                    line += 1

                by_line.append((line, -x, reverse_labels[a]))

            # sort by line then by x
            symbols_sorted = [(line, a) for line, x, a in sorted(by_line)]

            # print output to txt file with corresponding name
            txt_name = file[:file.rfind('.')] + ".txt"
            out_path = os.path.join(prediction_dir, txt_name)
            prev_line = 1
            with open(out_path, "w") as txt_file:
                for line, a in symbols_sorted:
                    if line != prev_line:
                        prev_line = line
                        txt_file.write("\n")
                    txt_file.write(a)

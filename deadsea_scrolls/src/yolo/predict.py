from util.global_params import *
import os
import numpy as np
from ultralytics import YOLO
import util.helper_functions as hf

reverse_labels = {
    0: 'Alef',
    1: 'Ayin',
    2: 'Bet',
    3: 'Dalet',
    4: 'Gimel',
    5: 'He',
    6: 'Het',
    7: 'Kaf',
    8: 'Kaf-final',
    9: 'Lamed',
    10: 'Mem',
    11: 'Mem-medial',
    12: 'Nun-final',
    13: 'Nun-medial',
    14: 'Pe',
    15: 'Pe-final',
    16: 'Qof',
    17: 'Resh',
    18: 'Samekh',
    19: 'Shin',
    20: 'Taw',
    21: 'Tet',
    22: 'Tsadi-final',
    23: 'Tsadi-medial',
    24: 'Waw',
    25: 'Yod',
    26: 'Zayin'}


def predict(file_folder):

    # setup the model
    model = YOLO(PREDICTION_MODEL)

    # check if results directory exists
    hf.assert_dir("results")

    # delete and recreate current if it already exists
    prediction_dir = os.path.join(PREDICTION_DIR, "current")
    if os.path.exists(prediction_dir):
        hf.remove_directory(prediction_dir)

    # create current dir
    os.makedirs(prediction_dir)

    # predict all binarized images and print output to a txt file
    for file in os.listdir(file_folder):

        if "binarized" in file:

            # do the prediction
            file_path = os.path.join(file_folder, file)
            boxes = model.predict(file_path)[0].boxes.boxes.cpu().detach().numpy()

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

                by_line.append((line, x, reverse_labels[a]))

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
                    txt_file.write(a + " ")

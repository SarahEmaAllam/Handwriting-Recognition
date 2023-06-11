import cv2
from PIL import Image
import numpy as np

Box = [float, float, float, float]
FOLDER = 'train'
WORD_LENGTH = 10
TEXT_LENGTH = 100 * np.random.randint(1, 5, size=1)[0]
# text_len = 10
# SCRIPT_SIZE = 608
NGRAM_SIZE = 4
WIDTH = 640
HEIGHT = 640
PADDING = 10 * np.random.randint(10, size=1)[0]
WHITESPACE = 15
SCRIPT_NAME = 'test'


def save_coco_label(points: Box):
    """
    Saves the true_label of the image in coco format: classs, x_c, y_c, w, h
    """
    x_c = points[0]
    y_c = points[1]
    w = points[2]
    h = points[3]
    label = '{} {} {} {}'.format(x_c, y_c, w, h).replace('"', '')
    file = str('labeltest') + '.txt'
    with open(str(file), 'a') as f:
        f.write(label)
        f.write("\n")
        f.close()


def draw_boxes(img, x, y, w, h):
    dh, dw, _ = np.asarray(img).shape
    l = int((x - w / 2) * dw)
    r = int((x + w / 2) * dw)
    t = int((y - h / 2) * dh)
    b = int((y + h / 2) * dh)

    if l < 0:
        l = 0
    if r > dw - 1:
        r = dw - 1
    if t < 0:
        t = 0
    if b > dh - 1:
        b = dh - 1

    image = cv2.rectangle(np.asarray(img), (l, t), (r, b), (255, 0, 0), 1)
    cv2.imshow('image', image)
    # Displaying the image
    cv2.waitKey(0)


img = cv2.imread('script-0.jpeg', cv2.IMREAD_UNCHANGED)
im = Image.fromarray(img)

with open('script-0.txt') as f:
    lines = f.readlines()
    for line in lines:
        print(line)
        f_line = []
        for x in line.split(" "):
            f_line.append(float(x))
        print(f_line)
        line = f_line
        x_c = line[1]
        print(x_c)
        y_c = line[2]
        w = line[3]
        print(w)
        h = line[4]
        box = [x_c, y_c, w, h]
        draw_boxes(im, x_c, y_c, w, h)
        save_coco_label(box)
# true_label = text[idx]

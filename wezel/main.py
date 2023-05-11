import os
import cv2
import numpy as np


def detect_on_edge(labels, amount):
    length, width = labels.shape
    edgelabels = np.zeros(amount)
    for i in range(width):
        edgelabels[labels[0][i]] = 1
        edgelabels[labels[length - 1][i]] = 1

    for i in range(length):
        edgelabels[labels[i][0]] = 1
        edgelabels[labels[i][width - 1]] = 1

    return edgelabels


def component_size(labels, amount):
    labelsize = np.zeros(amount)
    for iy, ix in np.ndindex(labels.shape):
        labelsize[labels[iy][ix]] = labelsize[labels[iy][ix]] + 1

    return labelsize


def remove_components(labelsize, edgelabels, labels, img, sizelimit):
    for iy, ix in np.ndindex(img.shape):
        label = labels[iy][ix]
        if labelsize[label] < sizelimit and edgelabels[label] == 1:
            img[iy][ix] = 0
    return img


if __name__ == '__main__':
    userdir = "/home/s2390590/Desktop/hwr/monkbrill/" # input("give the folder of the train data:")
    outputdir = "/home/s2390590/Desktop/hwr/test/" # input("give the output folder:")
    closingkernel = np.array([[0, 0, 0, 1, 0, 0, 0],
                              [0, 1, 1, 1, 1, 1, 0],
                              [1, 1, 1, 1, 1, 1, 1],
                              [0, 1, 1, 1, 1, 1, 0],
                              [0, 0, 0, 1, 0, 0, 0]], np.uint8)

    openingkernel = np.array([[0, 0, 1, 0, 0],
                              [0, 1, 1, 1, 0],
                              [0, 1, 1, 1, 0],
                              [1, 1, 1, 1, 1],
                              [0, 1, 1, 1, 0],
                              [0, 1, 1, 1, 0],
                              [0, 0, 1, 0, 0]], np.uint8)

    # kernel = np.array([[0, 1, 0],
    #                    [1, 1, 1],
    #                    [0, 1, 0]], np.uint8)

    for subdir, _, files in os.walk(userdir):
        fulloutpath = outputdir + os.path.basename(subdir) + "/"
        if not os.path.exists(fulloutpath):
            os.makedirs(fulloutpath)
        for file in files:
            filepath = os.path.join(subdir, file)
            srcimage = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            blurred = cv2.GaussianBlur(srcimage, (5, 5), 0)
            thresholded = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1] #  
#            closing = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, closingkernel)
            opening = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, openingkernel)
            amount, labels = cv2.connectedComponents(opening, 4, cv2.CV_32S)
            outputimg = opening
            if amount > 2:
                labelsize = component_size(labels, amount)
                edgelabels = detect_on_edge(labels, amount)
                outputimg = remove_components(labelsize, edgelabels, labels, opening, 200)

            cv2.imwrite(fulloutpath + file, cv2.bitwise_not(outputimg))

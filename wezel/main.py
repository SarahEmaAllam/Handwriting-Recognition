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
            img[iy][ix] = 255
    return img


def crop_clean_symbols(inputdir, outputdir):

    max_width = 0
    max_height = 0

    for subdir, _, files in os.walk(inputdir):
        print("Start crop clean: " + os.path.basename(subdir))
        fulloutpath = outputdir + "/" + os.path.basename(subdir) + "/"

        assert_dir(fulloutpath)

        for file in files:
            filepath = os.path.join(subdir, file)
            srcimage = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            blurred = cv2.GaussianBlur(srcimage, (3, 3), 0)
            thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            amount, labels, stats, _ = cv2.connectedComponentsWithStats(thresholded, 4, cv2.CV_32S)
            largest_label = 1
            if amount > 2:
                for label in range(1, amount):
                    if stats[largest_label, cv2.CC_STAT_AREA] < stats[label, cv2.CC_STAT_AREA]:
                        largest_label = label

            largest_component = stats[largest_label]
            left = largest_component[cv2.CC_STAT_LEFT]
            up = largest_component[cv2.CC_STAT_TOP]
            width = largest_component[cv2.CC_STAT_WIDTH]
            right = left + width
            height = largest_component[cv2.CC_STAT_HEIGHT]
            down = up + height
            cropped_img = srcimage[up:down, left:right]

            # keep track of the max bounds of the bounding boxes
            max_width = max(max_width, width)
            max_height = max(max_height, height)

            cropped_thresholded = cv2.threshold(cropped_img, 200, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_BINARY)[1]
            cropped_amount, cropped_labels, cropped_stats, _ = cv2.connectedComponentsWithStats(cropped_thresholded, 4, cv2.CV_32S)
            if cropped_amount > 2:
                label_size = cropped_stats[:, cv2.CC_STAT_AREA]
                edge_labels = detect_on_edge(cropped_labels, cropped_amount)
                cropped_img = remove_components(label_size, edge_labels, cropped_labels, cropped_img, 200)

            cv2.imwrite(fulloutpath + file, cropped_img)

        print("Finished crop clean: " + os.path.basename(subdir))

    return max_width, max_height


def rescale_images(imgdir, max_width, max_height):

    for subdir, _, files in os.walk(imgdir):
        print("Start rescaling: " + os.path.basename(subdir))
        for file in files:
            filepath = os.path.join(subdir, file)
            srcimage = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            height, width = srcimage.shape
            top = (max_height - height) // 2
            bottom = (max_height - height + 1) // 2
            left = (max_width - width) // 2
            right = (max_width - width + 1) // 2
            outputimg = cv2.copyMakeBorder(srcimage, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[255, 255, 255])
            cv2.imwrite(filepath, outputimg)

        print("Finished rescaling: " + os.path.basename(subdir))


def pre_processing(inputdir, outputdir):

    kernel = np.ones((6, 6), np.uint8)

    for file in os.listdir(inputdir):
        filepath = os.path.join(inputdir, file)
        if "binarized" in file:
            print("Start cleaning: " + file)
            srcimg = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            thresholded = cv2.threshold(srcimg, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            opening = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)
            closing = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)

            opendifference = cv2.bitwise_xor(thresholded, opening)
            closedifference = cv2.bitwise_xor(thresholded, closing)

            outputimg = srcimg
            for iy, ix in np.ndindex(outputimg.shape):
                # if opendifference[iy][ix]:
                #    outputimg[iy][ix] = 255

                if closedifference[iy][ix]:
                    outputimg[iy][ix] = 0

            cv2.imwrite(outputdir + "/" + file[0:-14] + ".pgm", outputimg)
            print("Finished cleaning: " + file)


def assert_dir(assrtdir):
    if not os.path.exists(assrtdir):
        os.makedirs(assrtdir)


if __name__ == '__main__':
    sourcesymboldir = input("give the folder of the train data:")
    sourcescollsdir = input("give the folder of the scrolls data:")
    outputdir = input("give the output folder:")

    assert_dir(outputdir)

    symbolsdir = outputdir + "/symbols/"
    assert_dir(symbolsdir)

    scrollsdir = outputdir + "/scrolls/"
    assert_dir(scrollsdir)

    print("Starting cropping and cleaning process.")
    maxWidth, maxHeight = crop_clean_symbols(sourcesymboldir, symbolsdir)
    print("Finished cropping and cleaning process")
    print("starting rescaling process")
    rescale_images(outputdir, maxWidth, maxHeight)

    print("Starting processing scrolls.")
    pre_processing(sourcescollsdir, scrollsdir)
    print("Finished processing scrolls.")





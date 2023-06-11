# Handwriting-Recognition

## Dead Sea Scrolls

1. Install the required packages.
Run the following command in the terminal, under the `deadsea_scrolls` directory preferably in a venv:
```
pip install -r requirements.txt
```

From here there are 4 options. 3 are connected to training and
1 is connected to testing.

### for training:

1. Add the dataset folder to the project directory (not included in the repository). 
The dataset folder should contain the following files:
- ```deadsea_scrolls/data/monkbrill``` - the training data (the letters).

1. Now from the `deadsea_scrolls` directory run we can choose to run 3 commands:

- ```python3 src/main.py all```:
This will preprocess all the monkbrill images if this isn't done yet,
generates the training data, train the model and tests it on the testing data.
The training and testing will start from scratch meaning if previously generated data
still exists it will be removed!

- ```python3 src/main.py train```:
This will only train model and try to continue if a previous run exists.

- ```python3 src/main.py generate```:
This will only generate the training data. This will remove the old training data

### for testing:

1. By adding a path to a folder: ```python3 src/main.py "path/to/image folder"``` instead of the previously mentioned commands the
program will try to do a prediction on all the images in the folder with `binarized` 
in their name. It will output the prediction in `results/current` as `.txt` files with
the name of the input image. If current already exists it's deleted and recreated.
Note: if the path contains a space surround it with `""`.

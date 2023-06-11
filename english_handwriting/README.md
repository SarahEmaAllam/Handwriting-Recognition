# Handwriting-Recognition

## IAM-dataset

1. Install the required packages.
Run the following command in the terminal, under the `english_handwriting` directory preferably in a venv:
```
pip install -r requirements.txt
```

### for training:
1. Add the dataset folder to the project directory (not included in the repository).
The dataset folder should contain the following files:
- ```english_handwriting/data/img/``` - the folder with the images.
- ```english_handwriting/data/IAM-data/iam_lines_gt.txt``` - the labels 
(file containing image filename and label for each image).
2. Go to `english_handwriting/` in the terminal and run `src/training.py`:
```
python src/training.py
```
If the preprocessing hasn't been done yet it will be done now.
Otherwise, it will load the datset (split into training, validation and test) and train the model.

3. You can evaluate the model in the `testing.py` file under the same directory.
The main function calls the `evaluate` function which loads the model and the test dataset and evaluates the model.
Note: make sure to specify in the `evaluate` function the path to the model you want to evaluate.
The paths to the models are specified in the `training.py` file (`logs/trained_models{other_info}/model_name`).

You can also run in the terminal, under the `english_handwriting` directory
the following command to see the loss of the model during training:
```
tensorboard --logdir logs/fit/
```

### for testing:
After installing the requirments.txt file, you can run the `main.py` file under the `english_handwriting` directory
as it follows:
```
python src/main.py <path_to_image>
```
Please note that the path to the image should be relative to the `english_handwriting` directory.
We suggest adding the folder with the test images directly under the `english_handwriting` directory.
Then you can just use `<images_folder_name>` as the path to the image.
# Handwriting-Recognition

## How to run the model

1. Install the required packages.
Run the following command in the terminal, under the project directory:
```
pip install -r requirements.txt
```

2. Add the dataset folder to the project directory (not included in the repository). 
The dataset folder should contain the following files:
- ```data/monkbrill``` - the training data (the letters);
- ```data/image-data``` - the testing data. If the testing set directory has a 
different name, you can keep it, but make sure to update the ```SOURCE_SCROLLS```
global variable in the ```globaL_params.py``` file.

3. Preprocessing the data - only needs to be done once. You can run the 
```preprocessing/main.py``` file:
```
python preprocess/preprocessing.py
```

5. Ultimately, to run the model, run the ```model.py``` file. It generates the 
training data, trains the model and tests it on the testing data. You can also 
uncomment the first line of code from the main function (and comment the others)
to generate a sample image of the training data.

```
python model.py
```

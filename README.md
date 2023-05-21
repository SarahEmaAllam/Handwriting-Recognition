# Handwriting-Recognition

## How to run the model

1. Install the required packages
Run the following command in the terminal, under the project directory:
```
pip install -r requirements.txt
```

2. Add the dataset folder to the project directory (not included in the repository). 
The dataset folder should contain the following files:
```data/monkbrill, data/image-data```, where ```data/image-data``` contains the testing data.

3. Preprocessing the data - if you wish to preprocess the data, run the ```preprocessing/main.py``` file

4. To generate training data, run the ```generate_data/generator_text.py``` file.

Note: the ```ngrams_frequencies_withNames_fixed``` is already in the generate_data directory, so you don't need to run the fix.py file.

Note2: you can see the character level augmentations in the Jupyter notebook ```andreea/char_aug.ipynb```  
with several examples of how each transformation works.

import pandas as pd
import os
from util.global_params import N_GRAM_PATH


filename = '../resources/ngrams_frequencies_withNames_prob.xlsx'
filepath = os.path.join(N_GRAM_PATH, filename)
df = pd.read_excel(filepath)
text = df["Names"]
text = text.str.replace('Tsadi', 'Tsadi-medial')
text = text.str.replace('Tasdi-final', 'Tsadi-final')
df["Names"] = text
new_filename = '../resources/ngrams_frequencies_withNames_fixed.xlsx'
new_filepath = os.path.join(N_GRAM_PATH, new_filename)
df.to_excel(new_filepath, index=False)

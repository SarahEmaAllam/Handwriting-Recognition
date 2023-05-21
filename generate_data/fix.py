import pandas as pd

path = 'ngrams_frequencies_withNames_prob.xlsx'
df = pd.read_excel(path)
text = df["Names"]
text = text.str.replace('Tsadi', 'Tsadi-medial')
text = text.str.replace('Tasdi-final', 'Tsadi-final')
df["Names"] = text
df.to_excel('ngrams_frequencies_withNames_fixed.xlsx', index=False)

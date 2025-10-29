from __future__ import unicode_literals, print_function, division
import contractions
import pandas as pd
from backend.constants.constant import MAX_LENGTH



def contractions_func(v : str) -> str:
    return contractions.fix(v)

# Read the csv file.
df2 = pd.read_csv("backend/data/raw/english_to_bangla.csv", encoding='utf-8')

df1 = pd.read_csv("backend/data/raw/eng_to_bng.csv", encoding='utf-8') 

print(f"All the datas are collected from source files")

df = pd.concat([df1, df2], ignore_index=True)

df["en"] = df["en"].apply(contractions_func)

df['en_word_length'] = df['en'].str.split().str.len()

df['bn_word_length'] = df['bn'].str.split().str.len()

df = df[~((df["en_word_length"] >=  MAX_LENGTH) | (df["bn_word_length"] >= MAX_LENGTH))]
df = df.reset_index(drop=True)

df = df[["en", "bn"]]

# Save the data into a csv file.
df.to_csv("backend/data/cleaned_data/cleaned_eng_bng.csv", index=False)
print(f"Cleaned data is saved in cleaned_eng_bng.csv file")


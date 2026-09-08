import pandas as pd

df = pd.read_csv('news_data_big.csv')

print(df.isnull().sum())


print("Shape:", df.shape)


print(df["label"].value_counts())



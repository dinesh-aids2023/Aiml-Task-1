
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler


df = pd.read_csv("Titanic-Dataset.csv")  
print("First 5 rows:\n", df.head())
print("\nInfo:\n")
df.info()
print("\nSummary statistics:\n", df.describe())
print("\nMissing values:\n", df.isnull().sum())

for col in df.select_dtypes(include=np.number).columns:
    df[col].fillna(df[col].mean(), inplace=True)

for col in df.select_dtypes(include="object").columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

df = pd.get_dummies(df, drop_first=True)  

scaler = StandardScaler()
num_cols = df.select_dtypes(include=np.number).columns
df[num_cols] = scaler.fit_transform(df[num_cols])



plt.figure(figsize=(15, 6))
df[num_cols].boxplot()
plt.xticks(rotation=90)
plt.title("Boxplot for Numerical Features")
plt.show()

from scipy.stats import zscore
z_scores = np.abs(zscore(df[num_cols]))
df = df[(z_scores < 3).all(axis=1)]

print("Cleaned Data Shape:", df.shape)

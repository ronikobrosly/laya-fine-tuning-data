import pandas as pd
import json

df = pd.read_parquet("typed_decisions_all.parquet")

# Overview
print(df.shape)               # rows, columns
print(df.dtypes)              # column types
df.head()                     # first 5 rows

# How the data splits up
df["workflow"].value_counts()
df["split"].value_counts()
pd.crosstab(df["workflow"], df["split"])
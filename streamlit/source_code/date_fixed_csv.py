import pandas as pd
import os


df = pd.read_csv('../Financial_Diaries_Modified.csv')
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df = df.drop(columns=['Formatted_Date'], axis=1)
df.to_csv('Financial_Diaries_final1.csv', index=False)
print(df)
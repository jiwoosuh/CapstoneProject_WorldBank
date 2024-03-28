import os
import pandas as pd
from pathlib import Path

os.chdir(Path(os.getcwd()).parent)
df = pd.read_csv('Financial_Diaries.csv')

from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# def classify_transaction(transaction_name):
#     # Candidate labels based on the provided categories
#     candidate_labels = [
#         'Business',
#         'Agriculture',
#         'Travel',
#         'Gifts',
#         'Household',
#         'Consumables',
#         'Financial Management',
#         'Health Care',
#         'Personal',
#         'Miscellaneous'
#     ]
#     result = classifier(transaction_name, candidate_labels)
#     return result['labels'][0]

def classify_transaction(transaction_name):
    # Candidate labels based on the provided categories
    candidate_labels = [
        'Miscellaneous',
        'Woman Personal',
        'Children',
        'Investment',
        'Gift',
        'Household',
        'Health Care',
        'Business',
        'Agriculture'
    ]
    result = classifier(transaction_name, candidate_labels)
    return result['labels'][0]


df1 = df.iloc[:50]

df1['Transaction_Category1'] = df1['Transaction_Name'].apply(classify_transaction)

print(df1[['Transaction_Name', 'Transaction_Category1'].head(10))




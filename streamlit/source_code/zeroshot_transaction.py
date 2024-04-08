import os
import pandas as pd
from pathlib import Path
from datasets import Dataset
import torch

os.chdir(Path(os.getcwd()).parent)
df = pd.read_csv('Financial_Diaries.csv')

from transformers import pipeline

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device)

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



# def classify_transaction(transaction_name):
#     labels = [
#         'Miscellaneous',
#         'Woman Personal',
#         'Children',
#         'Investment',
#         'Gift',
#         'Household',
#         'Health Care',
#         'Business',
#         'Agriculture'
#     ]
#     result = classifier(transaction_name, labels)
#     return result['labels'][0]

def classify_transaction(df):
    labels = [
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
    dataset = Dataset.from_pandas(df)
    result = classifier(dataset['Transaction_Name'], labels)
    return result


# df1 = df.iloc[:50]

# df['Transaction_Category1'][:10] = df['Transaction_Name'][:10].apply(classify_transaction)

print(classify_transaction(df))
# print(df[['Transaction_Name','Transaction_Category1']].head(10))

# df.to_csv('Financial_Diaries_withZeroShot.csv')

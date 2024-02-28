import os
import pandas as pd
from pathlib import Path

os.chdir(Path(os.getcwd()).parent)
df = pd.read_csv('Financial_Diaries.csv')

from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def classify_transaction(transaction_name):
    # Candidate labels based on the provided categories
    candidate_labels = [
        'Business',
        'Agriculture',
        'Travel',
        'Gifts',
        'Household',
        'Consumables',
        'Financial Management',
        'Health Care',
        'Personal',
        'Miscellaneous'
    ]
    result = classifier(transaction_name, candidate_labels)
    return result['labels'][0]

df['Transaction_Category'] = df['Transaction_Name'].apply(classify_transaction)

print(df[['Transaction_Name', 'Transaction_Category']].head(10))




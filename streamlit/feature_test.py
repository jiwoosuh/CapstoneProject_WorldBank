from transformers import pipeline
import pandas as pd


def zeroshot_transaction(df):

    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    def classify_transaction(df_col):
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
        result = classifier(df_col, labels)
        return result['labels'][0]

    df['Transaction_Category1'] = classify_transaction(df['Transaction_Name'])
    # print(classified_df[['Transaction_Name', 'Transaction_Category1']].head(10))
    return df


df = pd.read_csv("Financial_Diaries.csv")
df['Transaction_Category1'] = zeroshot_transaction(df["Transaction_Name"])
print(df['Transaction_Category1']).head(10)
import pandas as pd
import os
import re
from datetime import datetime

os.getcwd()
df = pd.read_csv('combined_output.csv')
print(df.head())

#%%
# Data Cleaning for date format
def clean_date_format(date_str):
    date_str = re.sub(r'\n[^\d]*', '', date_str)
    #date_str = re.sub(r'(\d{1,2})[^\d]*(\d{1,2})[^\d]*(\d{2}|\d{4})[^\d]*', r'\1/\2/\3', date_str)
    match = re.match(r'(\d{1,2})(?:st|nd|rd|th|-|)? (\w+) (\d+)', date_str, re.IGNORECASE)

    if match:
        day, month, year = match.groups()
        # if len(year) == 2:
        #     # Convert two-digit year to four-digit year
        #     current_year = datetime.now().year
        #     prefix = current_year // 100
        #     year = prefix * 100 + int(year)
        if month.isnumeric():
            month_number = int(month)
        else:
            # convert month to numeric
            month_number = datetime.strptime(month, '%B').month

        formatted_date = f'{day}/{month_number:02d}/{year}'
        fix_year_format(formatted_date)
        return formatted_date
    else:
        date_str = re.sub(r'(\d{1,2})[^\d]*(\d{1,2})[^\d]*(\d{2}|\d{4})[^\d]*', r'\1/\2/\3', date_str)
        return fix_year_format(date_str)


def fix_year_format(date_str):
    try:
        # Parse the date using the specified format
        formatted_date = datetime.strptime(date_str, '%d/%m/%y').strftime('%d/%m/%Y')
        # Format the date as dd/mm/yyyy
        return formatted_date
    except ValueError:
        # Handle the case where the date string is not in the expected format
        return date_str

# Clean member status into only two (WAG/NON WAG)
def clean_mem_status(mem_status):
    if 'NON WAG' in mem_status.upper():
        return 'NON WAG'
    else:
        return 'WAG'
# Clean member status into only two (WAG/NON WAG)
def clean_transaction_amount(amount_str):
    return pd.to_numeric(re.sub(r'[^\d.]', '', amount_str))


df['Formatted_Date'] = df['Date'].apply(clean_date_format)
print(set(df.Formatted_Date))
df['Transaction_Amount'] = df['Transaction_Amount'].apply(clean_transaction_amount)
df['Member_Status'] = df['Member_Status'].apply(clean_mem_status)
df['State'] = df['State'].str.lower()
df['State'] = df['State'].replace({'abia baseline': 'abia'})
df['Region'] = df['Region'].str.lower()
df['Transaction_Name'] = df['Transaction_Name'].str.replace('₦', '')

print(df.head())

df.to_csv('Financial_Diaries.csv', index=False)
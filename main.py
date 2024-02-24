import os
import csv
import pandas as pd
from source-code.word2csv import get_file_locations, extract_info_from_docx, convert_table_to_csv_file
from source-code.data_cleaning import clean_date_format, fix_year_format, clean_mem_status, clean_transaction_amount

#%%
# Folder to search for files
folder = 'Data'

# Get file locations
file_locations = get_file_locations(folder)

# Initialize CSV file header
csv_file_header = ['FD_Name', 'State', 'Region', 'Member_Status', 'File_Name', 'Respondent ID', 'Date', 'Week', 'Transaction_Nature', 'Transaction_Type',  'Transaction_Name', 'Transaction_Amount', 'Transaction_Comment']

# Initialize the data for CSV file
combined_csv_data = []

# Convert each table to CSV data
for docx_file in file_locations:
    print(f'Processing file: {docx_file}')
    csv_data = convert_table_to_csv_file(docx_file, csv_file_header)
    combined_csv_data.extend(csv_data)

# Write the combined data to a CSV file
combined_output_csv = 'combined_output.csv'
with open(combined_output_csv, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(csv_file_header)  # Write the header
    writer.writerows(combined_csv_data)  # Write the data

print(f'Combined data saved to {combined_output_csv}')

#%%

os.getcwd()
df = pd.read_csv('combined_output.csv')
print(df.head())

df['Formatted_Date'] = df['Date'].apply(clean_date_format)
print(set(df.Formatted_Date))
df['Transaction_Amount'] = df['Transaction_Amount'].apply(clean_transaction_amount)
df['Member_Status'] = df['Member_Status'].apply(clean_mem_status)
df['State'] = df['State'].str.lower()
df['State'] = df['State'].replace({'abia baseline': 'abia'})
df['Region'] = df['Region'].str.lower()
df['Transaction_Name'] = df['Transaction_Name'].str.replace('₦', '')


df.to_csv('new.csv', index=False)
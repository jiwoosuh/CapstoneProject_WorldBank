from docx import Document
import csv
import os
from pathlib import Path

def process_docx_files(folder_path, output_csv):
    glob_pattern = '**/*.docx'
    filepath_gen = folder_path.glob(glob_pattern)

    combined_data = []

    for file in filepath_gen:
        doc_data = convert_table_a_to_list(file)
        combined_data.extend(doc_data)

    # Write the combined data to a single CSV file
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Transaction_Nature', 'Transaction_Type', 'Transaction_Name', 'Transaction_Amount', 'Transaction_Comment'])

        for row in combined_data:
            writer.writerow(row)

def convert_table_a_to_list(input_docx):
    # Read the Word document
    doc = Document(input_docx)

    # Initialize the data for CSV file
    csv_file_data = []

    for table in doc.tables:
        rows = table.rows
        n = len(rows)
        transaction_type = "Fixed"

        # Income section
        for i in range(n):
            transaction_nature = "Income"
            try:
                transaction_name = rows[i].cells[0].text.strip()
                transaction_amount = rows[i].cells[1].text.strip()
            except IndexError:
                print(f"Error processing Income section in {input_docx}: IndexError at row {i}")
                continue

            if transaction_name == "Variable weekly income":
                transaction_type = "Variable"

            if transaction_name not in ["Variable weekly income", "Fixed weekly income", "Total:", "Variable weekly income",
                                        "Variable weekly expenditure", "Comments", ""]:
                csv_file_data.append([transaction_nature, transaction_type, transaction_name, transaction_amount])

        # Expenditure section
        transaction_type = "Fixed"
        for i in range(n):
            transaction_nature = "Expenditure"
            try:
                transaction_name = rows[i].cells[2].text.strip()
                transaction_amount = rows[i].cells[3].text.strip()
                transaction_comment = rows[i].cells[4].text.strip()
            except IndexError:
                print(f"Error processing Expenditure section in {input_docx}: IndexError at row {i}")
                continue

            if transaction_name == "Variable weekly expenditure":
                transaction_type = "Variable"

            if transaction_name not in ["Variable weekly expenditure", "Fixed weekly expenditure", "Total:", "Total",
                                        "Variable weekly income", "Variable weekly expenditure", "Comments", ""]:
                csv_file_data.append([transaction_nature, transaction_type, transaction_name, transaction_amount, transaction_comment])

    return csv_file_data


def main():
    # Specify the folder containing the .docx files and the output CSV file path
    # data_folder = Path(os.getcwd()).parent / 'Financial_Diaries'
    data_folder = 'Code/input_dir'
    output_csv = 'combined_output.csv'

    # Process all .docx files in the specified folder and create a single CSV file
    process_docx_files(data_folder, output_csv)

if __name__ == "__main__":
    main()

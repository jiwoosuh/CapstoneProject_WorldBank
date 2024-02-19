from docx import Document
import csv


def convert_table_a_to_csv_file(input_docx, output_csv):
    # Read the Word document
    doc = Document(input_docx)

    # Define the header for CSV file
    csv_file_header = ['Transaction_Nature', 'Transaction_Type', 'Transaction_Name', 'Transaction_Amount',
                      'Transaction_Comment']

    # Initialize the data for CSV file
    csv_file_data = []

    for table in doc.tables:
        rows = table.rows
        n = len(rows)
        transaction_type = "Fixed"

        # Income section
        for i in range(n):
            transaction_nature = "Income"
            transaction_name = rows[i].cells[0].text.strip()  # The first column is transaction name
            transaction_amount = rows[i].cells[1].text.strip()  # The second column is transaction amount
            if transaction_name == "Variable weekly income":
                transaction_type = "Variable"

            # Skip specific content
            if transaction_name in ["Variable weekly income", "Fixed weekly income", "Total:", "Variable weekly income",
                                    "Variable weekly expenditure", "Comments", ""]:
                continue

            # Combine the extracted data into a row and add it to the CSV file data
            csv_file_data.append(
                [transaction_nature, transaction_type, transaction_name, transaction_amount])

        # Expenditure section
        transaction_type = "Fixed"
        for i in range(n):
            transaction_nature = "Expenditure"
            transaction_name = rows[i].cells[2].text.strip()  # The third column is transaction name
            transaction_amount = rows[i].cells[3].text.strip()  # The fourth column is transaction amount
            transaction_comment = rows[i].cells[4].text.strip()  # The fifth column is transaction comment
            if transaction_name == "Variable weekly expenditure":
                transaction_type = "Variable"

            # Skip specific content
            if transaction_name in ["Variable weekly expenditure", "Fixed weekly expenditure", "Total:", "Total",
                                    "Variable weekly income", "Variable weekly expenditure", "Comments", ""]:
                continue

            # Combine the extracted data into a row and add it to the CSV file data
            csv_file_data.append(
                [transaction_nature, transaction_type, transaction_name, transaction_amount, transaction_comment])

    # Write the data to a CSV file
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csv_file_header)  # Write the header
        writer.writerows(csv_file_data)  # Write the data


# Specify the input Word document and output CSV file paths
input_docx = '1example.docx'
output_csv = 'table_output.csv'

# Convert Table A to CSV file
convert_table_a_to_csv_file(input_docx, output_csv)
import re
import csv
from docx import Document


def extract_info_from_docx(docx_file):
    doc = Document(docx_file)
    text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
    respondent_id_match = re.search(r'(?:Respondent ID:\s*)?(BL.+)', text)
    date_match = re.search(r'Date:\s*(\d{1,2}(?:st|nd|rd|th)?\s*\w+\s*\d{4}|[\d/]+)', text)
    week_match = re.search(r'WK\s+(\d+)', text, re.IGNORECASE)
    wag_match = re.search(r'BL\d+-(\w+)-\w+', text)

    # Extract information if there are matches or return None
    respondent_id = respondent_id_match.group(1) if respondent_id_match else None
    date = date_match.group(1) if date_match else None
    week = week_match.group(1) if week_match else None
    wag = wag_match.group(1) if wag_match else None

    return respondent_id, date, week, wag

def save_info_to_csv(respondent_id, date, week, wag, output_csv_file):
    with open(output_csv_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Respondent ID', 'Date', 'Week', 'Member Status'])
        writer.writerow([respondent_id, date, week, wag])


# Extract information
respondent_id, date, week, wag = extract_info_from_docx('1example.docx')

# Save info to csv
save_info_to_csv(respondent_id, date, week, wag, 'output.csv')

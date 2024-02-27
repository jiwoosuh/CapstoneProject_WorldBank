import streamlit as st
import os
import csv
import pandas as pd
from pathlib import Path
from streamlit.components.v1 import components
from source_code.word2csv import get_file_locations, extract_info_from_docx, convert_table_to_csv_file
from source_code.data_cleaning import clean_date_format, fix_year_format, clean_mem_status, clean_transaction_amount
from source_code.pdf2csv import pdf_to_images,ocr_handwritten_text, get_list_of_files

def main():
    st.title("Capstone Project")
    st.subheader("Social Sustainability and Inclusion")
    st.write("Nigeria For Women Program Scale-up Project with World Bank")
    st.write("Team Members: Brooklyn Chen, Jiwoo Suh, Sanjana Godolkar")
    st.write("Trello board URL: [Trello Board](https://trello.com/b/ytzd5Ve7/dats6501-brooklyn-chen-sanjana-godolkar)")

    # Sidebar navigation
    page = st.sidebar.selectbox("Select a page", ["Project Overview", "Methodology","Data Preprocessing", "OCR Handwritten PDF", "Analysis", "Conclusion"])

    # Page content
    if page == "Project Overview":
        project_overview()
    elif page == "Methodology":
        methodology()
    elif page == "Data Preprocessing":
        data_preprocessing()
    elif page == "OCR Handwritten PDF":
        pdf_ocr()
    elif page == "Analysis":
        analysis()
    elif page == "Conclusion":
        conclusion()

def project_overview():
    st.header("Project Overview")
    ## 1. Introduction
    st.subheader("1. Introduction")
    st.markdown("""
    This capstone seeks to digitize the financial processes of the Women Affinity
    Groups (WAGs) under the Nigeria for Women Program Scale Up (NFWP-SU)
    project. By structuring digitized financial transaction data, this will improve the efficiency for WAGs to
    harness technology for savings, credit access, and overall economic empowerment
    in order to help low-income women.
    """)

    ## 2. Background and Context
    st.subheader("2. Background and Context")
    st.markdown("""
    WAGs have been pivotal in fostering women's financial and social capital in
    Nigeria. They provide a platform for savings, mutual lending, and skill
    development. Despite their success, these groups face challenges in manual
    financial transactions, which are time-consuming and prone to errors. The
    digitization of these groups promises to streamline operations and expand their
    impact.
    """)

    ## 3. Problem Description
    st.subheader("3. Problem Description")
    st.markdown("""
    The project addresses the need for an efficient, transparent, and scalable solution to
    manage the financial activities of WAGs. By digitizing financial process, the
    project aims to automate savings, loan repayments, and other financial
    transactions, thereby enhancing the operational efficiency and financial
    empowerment of the groups.
    """)

    ## 4. Objectives and Goals
    st.subheader("4. Objectives and Goals")
    st.markdown("""
    - Digitize unstructured raw data into analyzable data.
    - Develop a dashboard to visualize financial transactions for further insights.
    - Measure the impact of digitization on the efficiency of WAGs and the economic
      empowerment of its members.
    """)

def methodology():
    st.header("Methodology")

    # Tasks done
    st.subheader("Tasks Completed:")
    st.markdown("""
    - **Data Preprocessing:**
      We extracted text and tables from docx documents and restructured them into CSV files to analyze. 
      Also, data cleaning was completed to match the format of the features including date, state name, and member status.
    """)

    # Tasks in progress
    st.subheader("Tasks In Progress:")
    st.markdown("""
    - **Text Classification and Keyword Extraction:**
      We are trying to analyze the text information in the financial transaction data to develop insights into the data 
      by classifying them into several categories and extracting the main keywords.

    - **Handwriting Recognition and OCR:**
      We are trying to develop a computer vision model to recognize the handwriting financial transaction data from 7 PDF files 
      and include them in our combined CSV data.
    """)

def pdf_ocr():
    st.header("OCR Handwritten PDF")
    data_folder = Path(os.getcwd()) / 'Data'
    pdf_files = get_list_of_files(data_folder, '**/*.pdf', '.pdf')

    for pdf_file in pdf_files:
        st.write(f"Processing PDF: {pdf_file}")
        images = pdf_to_images(pdf_file)

        for idx, image in enumerate(images):
            # Display each page of the PDF
            st.image(image, caption=f"Page {idx + 1}", use_column_width=True)

            # Perform OCR on the image
            text = ocr_handwritten_text(image)

            # Display OCR results
            st.write(f"Page {idx + 1} OCR Result:\n{text}\n")


def data_preprocessing():
    st.header("Data Preprocessing")

    # Data extraction
    folder = 'Data'

    file_locations = get_file_locations(folder)
    csv_file_header = ['FD_Name', 'State', 'Region', 'Member_Status', 'File_Name', 'Respondent ID', 'Date', 'Week', 'Transaction_Nature', 'Transaction_Type', 'Transaction_Name', 'Transaction_Amount', 'Transaction_Comment']

    combined_csv_data = []
    for docx_file in file_locations:
        st.write(f'Processing file: {docx_file}')
        csv_data = convert_table_to_csv_file(docx_file, csv_file_header)
        combined_csv_data.extend(csv_data)

    combined_output_csv = 'combined_output.csv'
    with open(combined_output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csv_file_header)  # Write the header
        writer.writerows(combined_csv_data)  # Write the data

    st.write(f'Combined data saved to {combined_output_csv}')

    # Data cleaning
    os.getcwd()
    df = pd.read_csv('combined_output.csv')

    df['Formatted_Date'] = df['Date'].apply(clean_date_format)
    df['Transaction_Amount'] = df['Transaction_Amount'].apply(clean_transaction_amount)
    df['Member_Status'] = df['Member_Status'].apply(clean_mem_status)
    df['State'] = df['State'].str.lower()
    df['State'] = df['State'].replace({'abia baseline': 'abia'})
    df['Region'] = df['Region'].str.lower()
    df['Transaction_Name'] = df['Transaction_Name'].str.replace('₦', '')

    df.to_csv('Financial_Diaries.csv', index=False)
    st.success('Data cleaning is completed. Financial_Diaries.csv saved.')
    # Display processed data
    if os.path.exists('Financial_Diaries.csv'):
        st.subheader("Processed Data")
        df = pd.read_csv('Financial_Diaries.csv')
        st.dataframe(df)




def analysis():
    st.header("Analysis")
    st.subheader("Tableau Dashboard")
    # tableau_url = "https://your-tableau-dashboard-url"
    # components.iframe(tableau_url, height=800, scrolling=True)

def conclusion():
    st.header("Conclusion")
    # Add content for Conclusion page

if __name__ == "__main__":
    main()

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
    st.title("🇳🇬 Nigeria For Women Program Scale-up Project")
    st.subheader("GWU Data Science Capstone Project - 2024 Spring")
    st.write("With World Bank 🌐 Social Sustainability and Inclusion")
    st.write("👩‍💻👩‍💻👩‍💻: Brooklyn Chen, Jiwoo Suh, Sanjana Godolkar")
    st.write("Instructor: Abdi Awl")
    st.write("[Trello Board](https://trello.com/b/ytzd5Ve7/dats6501-brooklyn-chen-sanjana-godolkar)")

    # Sidebar navigation
    page = st.sidebar.selectbox("Select a page",
                                ["Project Overview", "Main Tasks", "Methodology",
                                 "Data Preprocessing", "OCR Handwritten PDF", "Transaction Analysis",
                                 "Visual Analysis", "Challenges","Conclusion", "Further Tasks"])

    # Page content
    if page == "Project Overview":
        project_overview()
    elif page == "Main Tasks":
        tasks()
    elif page == "Methodology":
        methodology()
    elif page == "Data Preprocessing":
        data_preprocessing()
    elif page == "OCR Handwritten PDF":
        pdf_ocr()
    elif page == "Transaction Analysis":
        transaction_analysis()
    elif page == "Visual Analysis":
        visualization()
    elif page == "Challenges":
        challenges()
    elif page == "Conclusion":
        conclusion()
    elif page == "Further Tasks":
        further_analysis()

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

def tasks():
    st.header("Main Tasks")

    tasks_info = """

    ### Data Preprocessing - Docx2CSV, RegEx
    - **Description:** This task involves preparing the raw data for analysis. Two methods are employed - using the `Docx2CSV` utility and employing Regular Expressions (RegEx).
    - **Objective:** To convert data from Word documents (docx) into a more structured format (CSV) using `Docx2CSV` and perform additional preprocessing using RegEx for tasks like changing date formats.
    - **Significance:** This step is crucial for transforming unstructured data into a format suitable for analysis, ensuring uniformity and cleanliness in the dataset.

    ### OCR Handwritten PDF
    - **Description:** Optical Character Recognition (OCR) is applied to extract text from handwritten PDF documents.
    - **Objectives:**
        - Implement basic OCR techniques for extracting text from images.
        - Utilize unstructured I/O for partitioning PDFs.
        - Use the `pdf2image` library combined with transformers for advanced OCR on PDFs.
    - **Significance:** OCR on handwritten PDFs helps convert non-digital information into a format suitable for analysis, contributing to a comprehensive dataset.

    ### Transaction NLP Data Analysis
    - **Description:** Natural Language Processing (NLP) techniques are applied to analyze transaction data.
    - **Objectives:**
        - Apply traditional methods like TF-IDF (Term Frequency-Inverse Document Frequency) and Naive Bayes (NB) for analysis.
        - Implement zero-shot classification using transformer models.
    - **Significance:** Extracting insights from transactional data involves advanced NLP techniques, providing a deeper understanding of the context and sentiment associated with transactions.
    """

    st.markdown(tasks_info)


def methodology():
    st.header("Methodology")

    methodology_info = """

    ### Data Preprocessing - Docx2CSV
    - **Description:** The `Docx2CSV` custome function is utilized for converting Word documents to CSV format.
    - **Components:**
        - **Docx package:** Utilized for reading and extracting text and table data from Word documents.
    - **Significance:** This method facilitates the extraction of tabular data from Word documents and prepares it for further analysis.

    ### Data Preprocessing - RegEx
    - **Description:** Regular Expressions are employed to manipulate and clean data, specifically for changing date formats.
    - **Components:**
        - **Changing date format:** RegEx is used to modify the date formats present in the dataset.
    - **Significance:** RegEx provides a powerful tool for pattern-matching and manipulation, crucial for ensuring consistency in date representations.

    ### OCR Handwritten PDF
    - **Description:** OCR on handwritten PDFs involves several stages, including basic OCR, unstructured I/O for partitioning, and advanced OCR using `pdf2image` and transformers.
    - **Components:**
        - **Basic OCR:** Initial extraction of text from images.
        - **Unstructured I/O:** Partitioning PDFs into manageable sections.
        - **pdf2image + transformers:** Utilizing a combination of libraries for advanced OCR.
    - **Significance:** This multi-step approach ensures accurate extraction of text from handwritten PDFs, overcoming challenges associated with unstructured data.

    ### Transaction NLP Analysis
    - **Description:** Transactional data is subjected to NLP analysis using both traditional methods and transformer-based models.
    - **Components:**
        - **Traditional method:** TF-IDF and Naive Bayes for baseline analysis.
        - **Zero-shot classification:** Leveraging transformer models for advanced NLP analysis.
    - **Significance:** The combination of traditional and modern NLP techniques provides a comprehensive understanding of the textual information in transaction data.

    ### Data Visualization + Web App
    - **Description:** Visualization tools like Tableau and Streamlit are employed for creating interactive data visualizations and web applications.
    - **Components:**
        - **Tableau:** Used for creating visually appealing and informative dashboards.
        - **Streamlit:** Utilized for building interactive web applications for data exploration.
    - **Significance:** Visualization enhances the interpretability of the data, while web apps provide an accessible interface for users to interact with the findings.
    """

    st.markdown(methodology_info)

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


def transaction_analysis():
    st.header("Transactions NLP Analysis")

def visualization():
    import streamlit as st
    import numpy as np
    import pandas as pd
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    from PIL import Image
    from io import BytesIO
    import seaborn as sns
    from scipy import stats

    st.header("Analysis")
    # st.subheader("Tableau Dashboard")
    # tableau_url = "https://your-tableau-dashboard-url"
    # components.iframe(tableau_url, height=800, scrolling=True)

    # Load the dataset
    df = pd.read_csv("Financial_Diaries.csv", na_values={'transaction_comment': ''},
                     usecols=lambda column: column != 'Date', keep_default_na=False)
    df = df[df['Transaction_Amount'] != 0]

    # Convert columns to appropriate data types
    df['FD_Name'] = df['FD_Name'].astype('category')
    df['Member_Status'] = df['Member_Status'].astype('category')
    df['Week'] = df['Week'].astype('category')
    df['Transaction_Nature'] = df['Transaction_Nature'].astype('category')
    df['Transaction_Type'] = df['Transaction_Type'].astype('category')
    df['Formatted_Date'] = pd.to_datetime(df['Formatted_Date'], format='%d/%m/%Y', errors='coerce')
    df['Formatted_Date'] = df['Formatted_Date'].dt.date

    # Dataset Information
    st.subheader("Dataset Information")
    st.write(f"Number of Rows: {df.shape[0]}")
    st.write(f"Number of Columns: {df.shape[1]}")

    # Display first 5 unique values for each column
    st.subheader("Data Structure")
    info_df = pd.DataFrame(columns=["Column Name", "Data Type", "Non-Null Count", "Unique Values"])
    for column in df.columns:
        info_series = df[column].describe()
        data_type = df[column].dtype
        unique_values = df[column].unique()[:5]
        info_df = info_df.append({"Column Name": column,
                                  "Data Type": data_type,
                                  "Non-Null Count": info_series["count"],
                                  "Unique Values": ", ".join(map(str, unique_values))},
                                 ignore_index=True)
    st.write(info_df)

    # Plot
    # Calculate mean, median, and mode
    mean_val = df["Transaction_Amount"].mean()
    median_val = df["Transaction_Amount"].median()
    mode_val = stats.mode(df["Transaction_Amount"])[0][0]

    # Plot histogram and density plot
    st.subheader('Histogram and Density Plot with Mean, Median, and Mode')
    plt.figure(figsize=(10, 6))
    sns.histplot(df["Transaction_Amount"], kde=True, color='skyblue')
    plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=1, label='Mean: {:.2f}'.format(mean_val))
    plt.axvline(median_val, color='green', linestyle='dashed', linewidth=1, label='Median: {:.2f}'.format(median_val))
    plt.axvline(mode_val, color='orange', linestyle='dashed', linewidth=1, label='Mode: {:.2f}'.format(mode_val))
    plt.title('Histogram and Density Plot of {}'.format("Transaction_Amount"))
    plt.xlabel('Data Values')
    plt.ylabel('Frequency / Density')
    plt.legend()
    combined_fig = plt.gcf()
    st.pyplot(combined_fig)

    st.write("Due to the right-skewed nature of the Transaction_Amount data, we employ log transformation to achieve a more symmetrical distribution. This transformation mitigates the impact of large values on the distribution, compresses the range of values, and stabilizes variance. Additionally, it promotes linearity in the relationship between variables, a desirable trait in statistical modeling and analysis.")

    df['Transaction_Amount_log'] = np.log(df['Transaction_Amount'])

    # Calculate mean, median, and mode
    mean_val = df["Transaction_Amount_log"].mean()
    median_val = df["Transaction_Amount_log"].median()
    mode_val = stats.mode(df["Transaction_Amount_log"])[0][0]

    # Plot histogram and density plot after log
    st.subheader('Histogram and Density of Transaction_Amount_log')
    plt.figure(figsize=(10, 6))
    sns.histplot(df["Transaction_Amount_log"], kde=True, color='skyblue')
    plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=1, label='Mean: {:.2f}'.format(mean_val))
    plt.axvline(median_val, color='green', linestyle='dashed', linewidth=1, label='Median: {:.2f}'.format(median_val))
    plt.axvline(mode_val, color='orange', linestyle='dashed', linewidth=1, label='Mode: {:.2f}'.format(mode_val))
    plt.title('Histogram and Density Plot of {}'.format("Transaction_Amount_log"))
    plt.xlabel('Data Values')
    plt.ylabel('Frequency / Density')
    plt.legend()
    combined_fig = plt.gcf()
    st.pyplot(combined_fig)


    st.subheader("Average Transaction Amount by Members Status")
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    # Filter data for Member_Status == "NON WAG" and plot boxplot
    non_wag_data = df[df["Member_Status"] == "NON WAG"]
    non_wag_data_grouped = non_wag_data.groupby("Transaction_Type")["Transaction_Amount_log"]

    boxprops_non_wag = dict(facecolor="lightblue", edgecolor="black")

    axes[0].boxplot([non_wag_data_grouped.get_group("Income"), non_wag_data_grouped.get_group("Expenditure")],
                    vert=True,
                    patch_artist=True,
                    labels=["Income", "Expenditure"],
                    boxprops=boxprops_non_wag)
    axes[0].set_title("NON WAG")

    # Filter data for Member_Status == "WAG" and plot boxplot
    wag_data = df[df["Member_Status"] == "WAG"]
    wag_data_grouped = wag_data.groupby("Transaction_Type")["Transaction_Amount_log"]

    # Plot boxplot for WAG
    boxprops_wag = dict(facecolor="lightpink", edgecolor="black")

    axes[1].boxplot([wag_data_grouped.get_group("Income"), wag_data_grouped.get_group("Expenditure")],
                    vert=True,
                    patch_artist=True,
                    labels=["Income", "Expenditure"],
                    boxprops=boxprops_wag)
    axes[1].set_title("WAG")

    # Add labels and titles
    for ax in axes:
        ax.set_ylabel("Transaction Amount")
        ax.set_xlabel("Transaction Type")
        ax.yaxis.grid(True)

    # Display the plot using Streamlit
    st.pyplot(fig)


    # Word Cloud
    st.subheader("Word Cloud for Transaction Names")
    custom_stopwords = ["the", "and", "to", "of", "in", "for", "on", "with", "by", "from", "at", "is", "are", "was",
                        "were", "it", "that", "this", "an", "as", "or", "be", "have", "has", "not", "no", "can",
                        "could", "but", "so", "if", "when", "where", "how", "why", "which", "cost", "income", "weekly"]

    transaction_names = df['Transaction_Name'].str.lower().str.split()
    transaction_names = [[word for word in words if word not in custom_stopwords] for words in transaction_names]
    transaction_names_str = ' '.join([' '.join(words) for words in transaction_names])

    # Create the word cloud image
    wordcloud = WordCloud(width=800, height=800,
                          background_color='white',
                          min_font_size=10).generate(transaction_names_str)

    # Display the word cloud image using Streamlit
    st.image(wordcloud.to_array(), use_column_width=True)


def challenges():
    st.header("Challenge")

    challenge_info = """

    ### Communication with WB
    - **Description:** This challenge involves effective communication with the World Bank to shape and align the project with their expectations and requirements.
    - **Significance:** Ensuring clear communication helps in understanding and meeting the objectives set by the World Bank, leading to a successful collaboration.

    ### Unstructured Data
    - **Description:** The project deals with unstructured data in the form of MS Word docx files and handwritten PDF files.
    - **Challenges:**
        - **MS Word docx files:** Extraction of relevant information from unstructured Word documents.
        - **Handwritten PDF files:** OCR and extraction of text from handwritten PDFs.
    - **Significance:** Overcoming challenges related to unstructured data is crucial for obtaining valuable insights and ensuring data accuracy.

    ### Analysis & Deriving Insights
    - **Description:** The challenge involves the analysis of data and deriving meaningful insights from the processed information.
    - **Significance:** Analyzing the data and extracting insights contribute to the project's goals, providing valuable information for decision-making.

    """

    st.markdown(challenge_info)

def conclusion():
    st.header("Conclusion")
    # Add content for Conclusion page

def further_analysis():
    st.header("further_analysis")

if __name__ == "__main__":
    main()

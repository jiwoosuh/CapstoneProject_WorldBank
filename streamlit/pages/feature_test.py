import streamlit as st
import zipfile
import os
import shutil
import plotly.express as px
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from wordcloud import WordCloud
from plotly.subplots import make_subplots
import csv
from datasets import Dataset
import cv2
from pdf2image import convert_from_path
import easyocr
import torch
import sys
from pathlib import Path
import nltk
from nltk.corpus import stopwords
from collections import Counter
from transformers import pipeline
st.set_option('deprecation.showPyplotGlobalUse', False)
sys.path.append(Path(os.getcwd()).parent)

from source_code.word2csv import get_file_locations, extract_info_from_docx, convert_table_to_csv_file
from source_code.data_cleaning import clean_date_format, fix_year_format, clean_mem_status, clean_transaction_amount
from source_code.pdf2csv_easyOCR import ocr_result

st.set_page_config(
    layout='wide',
    page_title="Upload Data",
    page_icon="🪄"
)

st.markdown(
    f'<h1 style="background-color:#0169CA; color:white; text-align:center; border-radius: 5px;">Docx2Dashboard</h1>',
    unsafe_allow_html=True
)
# def extract_folder_name(zip_file):
#     extract_path = os.getcwd()
#
#     with zipfile.ZipFile(zip_file, 'r') as zip_ref:
#         zip_ref.extractall(extract_path)
#
#     macosx_folder = os.path.join(extract_path, "__MACOSX")
#     if os.path.exists(macosx_folder):
#         shutil.rmtree(macosx_folder)
#
# # @st.cache_data
# def data_extraction(folder):
#     file_locations = get_file_locations(folder)
#     num_files = len(file_locations)
#     csv_file_header = ['FD_Name', 'State', 'Region', 'Member_Status', 'File_Name', 'Respondent_ID', 'Date', 'Week', 'Transaction_Nature', 'Transaction_Type', 'Transaction_Name', 'Transaction_Amount', 'Transaction_Comment']
#     print(num_files)
#     combined_csv_data = []
#     progress_bar = st.progress(0, text = 'Extracting Data...')
#     for i, docx_file in enumerate(file_locations):
#         progress = (i + 1) / num_files
#         progress_bar.progress(progress, text = 'Extracting Data...')
#         #st.write(f'Processing file: {docx_file}')
#         csv_data = convert_table_to_csv_file(docx_file, csv_file_header)
#         combined_csv_data.extend(csv_data)
#
#     combined_output_csv = 'combined_output.csv'
#     with open(combined_output_csv, 'w', newline='', encoding='utf-8') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(csv_file_header)  # Write the header
#         writer.writerows(combined_csv_data)  # Write the data
#     return combined_output_csv
#
# def data_cleaning(combined_ouput_csv):
#     # Data cleaning
#     # os.getcwd()
#     df = pd.read_csv(combined_ouput_csv)
#
#     # df['Formatted_Date'] = df['Date'].apply(clean_date_format)
#     df['Date'] = pd.to_datetime(df['Date'])
#     df['Transaction_Amount'] = df['Transaction_Amount'].apply(clean_transaction_amount)
#     df['Member_Status'] = df['Member_Status'].apply(clean_mem_status)
#     df['State'] = df['State'].str.lower()
#     df['State'] = df['State'].replace({'abia baseline': 'abia'})
#     df['Region'] = df['Region'].str.lower()
#     df['Transaction_Name'] = df['Transaction_Name'].str.replace('₦', '')
#     replacement_dict = {
#         'obingwa': 'obi ngwa',
#         'ijebu ne': 'ijebu north east',
#         'maiyami': 'maiyama',
#         'ibesikpo': 'ibesikpo asutan'
#     }
#     df['Region'] = df['Region'].replace(replacement_dict)
#
#     # Convert columns to appropriate data types
#     df['FD_Name'] = df['FD_Name'].astype('category')
#     df['Member_Status'] = df['Member_Status'].astype('category')
#     df['Week'] = df['Week'].astype('category')
#     df['Transaction_Nature'] = df['Transaction_Nature'].astype('category')
#     df['Transaction_Type'] = df['Transaction_Type'].astype('category')
#     # df['Formatted_Date'] = pd.to_datetime(df['Formatted_Date'], format='%d/%m/%Y', errors='coerce')
#
#     # Remove 'zero Transaction_Amount' and 'Date' column
#     df = df[df['Transaction_Amount'] != 0]
#     # df.drop(columns=['Date'], inplace=True)
#     return df
#
#
# def add_column_if_missing(df, column_name, after_column):
#     """
#     Add a column to a DataFrame if it doesn't already exist, positioning it after a specified column.
#
#     Parameters:
#         df (DataFrame): The DataFrame to check and potentially modify.
#         column_name (str): The name of the column to be added if missing.
#         after_column (str): The name of the column after which the new column should be inserted.
#
#     Returns:
#         DataFrame: The modified DataFrame.
#     """
#     if column_name not in df.columns:
#         idx = df.columns.get_loc(after_column) + 1
#         df.insert(idx, column_name, None)
#     return df
#
# def zeroshot_transaction(df):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(device)
#     classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device)
#     def classify_transaction(transaction_name):
#         labels = [
#             'Miscellaneous',
#             'Woman Personal',
#             'Children',
#             'Investment',
#             'Gift',
#             'Household',
#             'Health Care',
#             'Business',
#             'Agriculture'
#         ]
#         result = classifier(transaction_name, labels)
#         return result['labels'][0]
#     df = add_column_if_missing(df, 'Transaction_Category1', 'Transaction_Name')
#     # df['Transaction_Category1'] = classify_transaction(df['Transaction_Name'])
#     df['Transaction_Category1'] = df['Transaction_Name'].apply(classify_transaction)
#     # print(classified_df[['Transaction_Name', 'Transaction_Category1']].head(10))
#     return df
#
# def data_update_and_save(old_data,new_data,file_name):
#     old_data = pd.read_csv(old_data)
#     updated_data = pd.concat([old_data,new_data])
#     updated_data = updated_data.drop_duplicates()
#     # st.download_button(
#     #     label="Download Updated CSV",
#     #     data=updated_data.to_csv(index=False).encode(),
#     #     file_name=file_name,
#     #     mime="text/csv"
#     # )
#
#     updated_data.to_csv(file_name, index=False)
#     st.success(f'Data is cleaned and saved as {file_name}')
#
#     # Display processed data
#     if os.path.exists(file_name):
#         st.subheader("Processed Data")
#         df = pd.read_csv(file_name)
#         st.dataframe(df)
#         return df
#
# def display_data_structure(df):
#     info_data = []
#     for column in df.columns:
#         data_type = df[column].dtype
#         non_null_count = df[column].count()
#         unique_values = df[column].unique()[:6]
#         info_data.append({"Column Name": column,
#                           "Data Type": data_type,
#                           "Non-Null Count": non_null_count,
#                           "Unique Values": ", ".join(map(str, unique_values))})
#     info_df = pd.DataFrame(info_data)
#
#     with st.expander("Expand to view"):
#         st.write(f"Number of Transactions: {df.shape[0]}")
#         st.write(f"Number of Variables: {df.shape[1]}")
#         st.dataframe(info_df)
#
# def display_overview(df):
#     # df['Formatted_Date'] = pd.to_datetime(df['Formatted_Date'])
#     # df['Year'] = df['Formatted_Date'].dt.year
#     df['Date'] = pd.to_datetime(df['Date'])
#     df['Year'] = df['Date'].dt.year
#
#     count_r = df['Respondent_ID'].nunique()
#     count_wag = df[df["Member_Status"] == "WAG"]['Respondent_ID'].nunique()
#     count_nwag = df[df["Member_Status"] == "NON WAG"]['Respondent_ID'].nunique()
#     total_income = df[df["Transaction_Type"] == "Income"]
#     income_per = (total_income["Transaction_Amount"].sum()/df["Transaction_Amount"].sum())*100
#     total_expence = df[df["Transaction_Type"] == "Expenditure"]
#     expence_per = (total_expence["Transaction_Amount"].sum()/df["Transaction_Amount"].sum())*100
#     total_fixed = df[df["Transaction_Nature"] == "Fixed"]
#     fixed_per = (total_fixed["Transaction_Amount"].sum()/df["Transaction_Amount"].sum())*100
#     total_var = df[df["Transaction_Nature"] == "Variable"]
#     var_per = (total_var["Transaction_Amount"].sum()/df["Transaction_Amount"].sum())*100
#
#     st.markdown(f"""
#     <div style='background-color: #CCE5FF; padding: 10px; border-radius: 10px;'>
#         <div style='text-align: center; display: flex; font-size: 18px;'>
#             <div>
#                 <div style='display: flex; align-items: center;'>
#                     <div style='margin: 15px;'>
#                         <p style='margin-bottom: 10px; font-weight: bold; font-size: 20px;'>Respondent</p>
#                         <div style='display: flex; justify-content: space-between; width: 500px; height:100px; background-color: white; padding: 20px; border-radius: 10px;'>
#                             <div style='margin: 0 5% 0 10%;'>Total<br>{count_r}</div>
#                             <div style='margin: 0 5% 0 5%;'>WAG<br>{count_wag}</div>
#                             <div style='margin: 0 10% 0 5%;'>NWAG<br>{count_nwag}</div>
#                         </div>
#                     </div>
#                     <div style='margin: 15px;'>
#                         <p style='margin-bottom: 10px;font-weight: bold; font-size: 20px;'>Transaction Type</p>
#                         <div style='display: flex; justify-content: space-between; width: 340px; height:100px; background-color: white; padding: 20px; border-radius: 10px;'>
#                             <div style='margin: 0 3% 0 10%;'>Income<br>{income_per.round(1)}%</div>
#                             <div style='margin: 0 8% 0 3%;'>Expenditure<br>{expence_per.round(1)}%</div>
#                         </div>
#                     </div>
#                     <div style='margin: 10px;'>
#                         <p style='margin-bottom: 10px;font-weight: bold; font-size: 20px;'>Transaction Nature</p>
#                         <div style='display: flex; justify-content: space-between; width: 340px; height:100px; background-color: white; padding: 20px; border-radius: 10px;'>
#                             <div style='margin: 0 3% 0 10%;'>Fixed<br>{fixed_per.round(1)}%</div>
#                             <div style='margin: 0 8% 0 3%;'>Variable<br>{var_per.round(1)}%</div>
#                         </div>
#                     </div>
#                 </div>
#             </div>
#         </div>
#     </div>
#     """, unsafe_allow_html=True)
#     wag_transaction_amount_log = np.log(df[df['Member_Status'] == 'WAG']['Transaction_Amount'])
#     non_wag_transaction_amount_log = np.log(df[df['Member_Status'] == 'NON WAG']['Transaction_Amount'])
#
#     # Create histogram traces
#     trace1 = go.Histogram(x=wag_transaction_amount_log, marker=dict(color='navy'), name='WAG')
#     trace2 = go.Histogram(x=non_wag_transaction_amount_log, opacity=0.7, marker=dict(color='skyblue'), name='NON WAG')
#
#     layout = go.Layout(
#         title='Distribution of Transaction Amount (log)',
#         xaxis=dict(title='Transaction Amount (log)'),
#         yaxis=dict(title='Frequency'),
#         barmode='overlay',
#     )
#     fig1 = go.Figure(data=[trace1, trace2], layout=layout)
#     fig1.update_layout(legend=dict(
#         orientation="h",
#         yanchor="bottom",
#         y=1.02,
#         xanchor="right",
#         x=1
#     ))
#
#     # Plot 2
#     fig2 = px.bar(df, x='Transaction_Type', y='Transaction_Amount', color='Transaction_Nature',
#                   title='Transaction Amount by Type and Nature')
#     fig2.update_layout(legend=dict(
#         orientation="h",
#         yanchor="bottom",
#         y=1.02,
#         xanchor="right",
#         x=1
#     ))
#
#     # Plot 3
#     fig3 = px.bar(df, x='Week', y='Transaction_Amount', color='Transaction_Type',
#                   title='Transaction Amount by Week')
#     fig3.update_layout(legend=dict(
#         orientation="h",
#         yanchor="bottom",
#         y=1.02,
#         xanchor="right",
#         x=1
#     ))
#     # Plot 4
#     fig4 = make_subplots(rows=1, cols=2, shared_yaxes=True)
#     fig4.add_trace(
#         go.Bar(x=total_income['Year'], y=total_income['Transaction_Amount']),
#         row=1, col=1
#     )
#
#     fig4.add_trace(
#         go.Bar(x=total_expence['Year'], y=total_expence['Transaction_Amount']),
#         row=1, col=2
#     )
#
#     fig4.update_traces(showlegend=False)
#
#     # Add subtitles
#     fig4.update_layout(
#         title_text="Transaction Amount by Year",
#         annotations=[
#             dict(
#                 text="Income", x=0.220, y=-0.15, showarrow=False, xref="paper", yref="paper", font=dict(size=14)
#             ),
#             dict(
#                 text="Expense", x=0.780, y=-0.15, showarrow=False, xref="paper", yref="paper", font=dict(size=14)
#             )
#         ]
#     )
#     fig4.update_layout(height=450, width=800, title_text="Transaction Amount by Year")
#
#     # Plot 5
#     custom_stopwords = ["the", "and", "to", "of", "in", "for", "on", "with", "by", "from", "at", "is", "are", "was",
#                         "were", "it", "that", "this", "an", "as", "or", "be", "have", "has", "not", "no", "can",
#                         "could", "but", "so", "if", "when", "where", "how", "why", "which", "cost", "income",
#                         "weekly"]
#
#     # Preprocess transaction names
#     transaction_names = df['Transaction_Name'].str.lower().str.split()
#     transaction_names = [[word for word in words if word not in custom_stopwords] for words in transaction_names]
#     transaction_names_str = ' '.join([' '.join(words) for words in transaction_names])
#
#     # Generate Word Cloud
#     wordcloud = WordCloud(background_color='white', colormap='Blues_r', height=600, width=1850,
#                           min_font_size=10).generate(transaction_names_str)
#
#     # Convert Word Cloud to Plotly image
#     img_rgb = wordcloud.to_array()
#     fig5 = go.Figure(data=go.Image(z=img_rgb))
#     fig5.update_layout(title='Word Cloud for Transaction Name',
#                        xaxis_visible=False,
#                        yaxis_visible=False)
#
#
#     # Plot 7
#     fig7 = px.bar(df, x='Week', y='Transaction_Amount', color='Transaction_Category1', barmode='group',
#                   title='Transaction Category by Week')
#
#     col2, col3, col4 = st.columns(3)
#     col1, col5 = st.columns([1, 2])
#
#     with col1:
#         st.plotly_chart(fig1, use_container_width=True)
#
#     with col2:
#         st.plotly_chart(fig2, use_container_width=True)
#
#     with col3:
#         st.plotly_chart(fig3, use_container_width=True)
#
#     with col4:
#         st.plotly_chart(fig4, use_container_width=True)
#
#     with col5:
#         st.plotly_chart(fig5, use_container_width=True)
#
#     st.plotly_chart(fig7, use_container_width=True)
#
#
# def preprocess_text(column):
#     nltk.download('punkt')
#     nltk.download('stopwords')
#     stop_words = set(stopwords.words('english'))
#     tokens = []
#     column = column.dropna()
#     for sent in column:  # Drop NaN rows
#         words = nltk.word_tokenize(sent.lower())
#         filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
#         tokens.extend(filtered_words)
#     return tokens
#
# def get_frequent_words(column, n=10):
#     all_words = preprocess_text(column)
#     word_counts = Counter(all_words)
#     frequent_words = word_counts.most_common(n)
#     return frequent_words
#
# def interactive_transaction_analysis(df, column_name, n):
#     frequent_words = get_frequent_words(df[column_name], n)
#     words = [pair[0] for pair in frequent_words]
#     frequencies = [pair[1] for pair in frequent_words]
#     # fig = go.Figure(go.Bar(x=words, y=frequencies, marker_color='skyblue'))
#     # fig.update_layout(xaxis=dict(title='Words'), yaxis=dict(title='Frequency'), title=f'Top {n} Frequent Words', xaxis_tickangle=-45, height=400, width=600)
#     # st.plotly_chart(fig)
#     fig = go.Figure(go.Bar(x=words, y=frequencies, marker_color='skyblue'))
#     fig.update_layout(xaxis=dict(title='Words'), yaxis=dict(title='Frequency'), title=f'Top {n} Frequent Words', xaxis_tickangle=-45, height=400, width=600)
#     return fig
#
#
# def get_unique_values(df):
#     unique_values_by_column = {}
#     for column in df.columns:
#         unique_values_by_column[column] = df[column].unique()
#     return unique_values_by_column
#
# def manual_update(old_data):
#     # Update Tabular Data Manually
#     if old_data is not None:
#         df = pd.read_csv(old_data)
#         unique_values = get_unique_values(df)
#
#         # Create select boxes for input variables based on unique values
#         st.subheader("Enter Variable Values:")
#         fd_name = st.text_input("FD_Name:")
#         state = st.selectbox("State:", unique_values['State'])
#         region = st.selectbox("Region:", unique_values['Region'])
#         member_status = st.selectbox("Member_Status:", unique_values['Member_Status'])
#         file_name = st.text_input("File_Name:")
#         respondent_id = st.text_input("Respondent ID:")
#         date = st.text_input("Date(DD/MM/YYYY):")
#         week = st.number_input("Week:", min_value = 1,max_value=5)
#         transaction_nature = st.selectbox("Transaction_Nature:", unique_values['Transaction_Nature'])
#         transaction_type = st.selectbox("Transaction_Type:", unique_values['Transaction_Type'])
#         transaction_category = st.selectbox("Transaction_Category:", unique_values['Transaction_Category'])
#         category_name = st.selectbox("Category_Name:", unique_values['Category_Name'])
#         transaction_name = st.text_input("Transaction_Name:")
#         transaction_amount = st.number_input("Transaction_Amount:")
#         transaction_comment = st.text_input("Transaction_Comment:")
#         # formatted_date = clean_date_format(date)
#
#         if st.button("Update Row"):
#             # Call functions to update tabular data
#             updated_row_data = [fd_name, state, region, member_status, file_name, respondent_id, date, week,
#                                 transaction_nature, transaction_type, transaction_category, category_name,
#                                 transaction_name, transaction_amount, transaction_comment]
#             # Update tabular data with the new row data
#             updated_data = update_tabular_data(df, updated_row_data)
#             st.success("Row Updated Successfully!")
#             st.dataframe(updated_data.tail())
#
#             updated_data_csv = updated_data.to_csv(index=False, encoding='utf-8')
#             st.download_button(
#                 label="Download data as CSV",
#                 data=updated_data_csv,
#                 file_name="Updated_data.csv",
#                 mime='text/csv',
#             )
#
# def update_tabular_data(old_data, updated_row_data):
#     new_row = pd.DataFrame([updated_row_data], columns=old_data.columns)
#     updated_data = pd.concat([old_data, new_row], ignore_index=True)
#     return updated_data
#
# def get_pdf_file_locations(folder):
#     pdf_file_locations = []
#     for root, dirs, files in os.walk(folder):
#         for file in files:
#             if file.endswith('.pdf'):  # Check if file ends with ".pdf"
#                 pdf_file_locations.append(os.path.join(root, file))
#     return pdf_file_locations
#
# def ocr_result(pdf_files):
#     for pdf_file in pdf_files:
#         st.write(f"Processing PDF: {pdf_file}")
#         pages = convert_from_path(pdf_file)
#         reader = easyocr.Reader(['en'], gpu=True)
#
#         # Loop through each page
#         for idx, page in enumerate(pages):
#             # Convert PIL image to OpenCV format
#             image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
#
#             # Sharpen the image
#             gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#             sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
#             sharpen = cv2.filter2D(gray, -1, sharpen_kernel)
#
#             # Thresholding
#             thresh = cv2.threshold(sharpen, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
#
#             # Perform OCR
#             ocr_results = reader.readtext(thresh, detail=0)
#
#             # Display OCR results
#             st.image(image, caption=f"Page {idx + 1}", use_column_width=True)
#             st.write(f"Page {idx + 1} OCR Result:\n{ocr_results}\n")
#
#             # Define an empty DataFrame
#             df = pd.DataFrame(
#                 columns=['FD_Name', 'State', 'Region', 'Member_Status', 'File_Name', 'Respondent_ID', 'Date', 'Week',
#                          'Transaction_Nature', 'Transaction_Type', 'Transaction_Category', 'Category_Name',
#                          'Transaction_Name', 'Transaction_Amount', 'Transaction_Comment'])
#             # df = pd.DataFrame(
#             #     columns=['FD_Name', 'State', 'Region', 'Member_Status', 'File_Name', 'Respondent_ID', 'Date', 'Week',
#             #              'Transaction_Nature', 'Transaction_Type', 'Transaction_Category', 'Category_Name',
#             #              'Transaction_Name', 'Transaction_Amount', 'Transaction_Comment', 'Formatted_Date'])
#
#             # Define column configurations
#             config = {
#                 'FD_Name': st.column_config.TextColumn('FD Name'),
#                 'State': st.column_config.TextColumn('State'),
#                 'Region': st.column_config.TextColumn('Region'),
#                 'Member_Status': st.column_config.TextColumn('Member Status'),
#                 'File_Name': st.column_config.TextColumn('File Name'),
#                 'Respondent_ID': st.column_config.TextColumn('Respondent ID'),
#                 'Date': st.column_config.DateColumn('Date'),
#                 'Week': st.column_config.NumberColumn('Week'),
#                 'Transaction_Nature': st.column_config.TextColumn('Transaction Nature'),
#                 'Transaction_Type': st.column_config.TextColumn('Transaction Type'),
#                 'Transaction_Category': st.column_config.TextColumn('Transaction Category'),
#                 'Category_Name': st.column_config.TextColumn('Category Name'),
#                 'Transaction_Name': st.column_config.TextColumn('Transaction Name'),
#                 'Transaction_Amount': st.column_config.NumberColumn('Transaction Amount'),
#                 'Transaction_Comment': st.column_config.TextColumn('Transaction Comment'),
#                 # 'Formatted_Date': st.column_config.TextColumn('Formatted Date')
#             }
#
#             # Generate a unique key based on PDF file name and page index for the data editor
#             editor_key = f"{pdf_file}_{idx}_editor"
#             # Generate a unique key based on PDFc file name and page index for the button
#             button_key = f"{pdf_file}_{idx}_button"
#
#             # Display the data editor and get user input
#             result = st.data_editor(df, column_config=config, num_rows='dynamic', key=editor_key)
#
#             # Button to get the results
#             if st.button('Get results', key=button_key):
#                 st.write(result)
#
# def process_data(folder_upload, old_upload, filename):
#     if folder_upload is not None:
#         if folder_upload.type == 'application/zip':
#             extract_folder_name(folder_upload)
#             folder = folder_upload.name[:-4]
#             combined_output = data_extraction(folder)
#             cleaned_data = data_cleaning(combined_output)
#             classified_data = zeroshot_transaction(cleaned_data)
#             final_output = data_update_and_save(old_data=old_upload, new_data=classified_data, file_name=filename)
#             pdf_files = get_pdf_file_locations(folder)
#             if pdf_files:
#                 st.info("PDF files found. Running OCR...")
#                 ocr_result(pdf_files)
#             return final_output
#         else:
#             st.error("Please upload a zip file.")

from utils import *

def main():
    # session_state = get_session_state()
    st.subheader("💡Unlock insights by uploading your documents")
    instruction="""
    **Upload Documents:**

    - Choose a zip file containing your documents.
    - Alternatively, if you want to update data manually, skip to step 2.
    
    **Choose a File to Update:**
    
    - The Docx2Dash will update the existing CSV file by default.
    - Optionally, you can upload your own CSV file.
    
    **Analyze Documents:**
    
    - Click "Docx2Dash" to analyze your documents.
    - Explore insights and summaries presented by the app.
    """
    st.markdown(instruction)
    folder_upload = st.file_uploader("Upload a zip file", type=["zip"], key="folder_upload")
    old_upload = 'Financial_Diaries_final.csv'
    use_own_csv = st.checkbox("Update your own CSV file")
    if use_own_csv:
        old_upload = st.file_uploader("Upload your CSV file", type=["csv"], key="old_upload")
    file_name = st.text_input("Enter the file name for updated data:", value='Financial_Diaries_updated.csv')
    if st.button("Docx2Dashboard"):
        final_output = process_data(folder_upload, old_upload, file_name)
        display_data_structure(final_output)
        display_overview(final_output)
        column_name = st.selectbox(
            'Choose between columns',
            ('Transaction_Name', 'Transaction_Comment'))
        n = st.sidebar.slider('Select number of top frequent words:', 1, 30, 10)

        # Placeholder for the visualization
        placeholder = st.empty()
        fig = interactive_transaction_analysis(final_output, column_name, n)
        placeholder.plotly_chart(fig)

    if st.button("Manual Update"):
        manual_update(old_upload)
        # manual_update(session_state['manual_update_data'])


if __name__ == "__main__":
    main()

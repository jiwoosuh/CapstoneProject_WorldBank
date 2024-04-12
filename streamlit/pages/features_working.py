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

from utils import *

def main():
    st.subheader("Unlock insights by uploading your documents 💡", divider = 'grey')
    st.subheader('1. What is your task?')
    update_type = st.radio(
        "Choose your main task",
        [
         "Update a current CSV file",
         "Update your own CSV file"
         ],
        label_visibility="collapsed"
    )
    if update_type == "Update a current CSV file":
        st.subheader('2. How do you want to update the current CSV file?')
        old_upload = 'Financial_Diaries_final.csv'
        task = st.radio(
            "Choose your update method",
            [
                "Upload new documents",
                "Type input manually"
            ],
            label_visibility="collapsed"
        )

        if task == "Upload new documents":
            folder_upload = st.file_uploader("Upload a zip file only", type=["zip"], key="folder_upload")
            # old_upload = 'Financial_Diaries_final.csv'

        if task == "Type input manually":
            manual_update(old_upload)

    if update_type == "Update your own CSV file":
        old_upload = st.file_uploader("Upload your previous CSV file", type=["csv"], key="old_upload")
        st.subheader('2. How do you want to update your own CSV file?')
        task = st.radio(
            "Choose your update method",
            [
                "Upload new documents",
                "Type input manually"
            ],
            label_visibility="collapsed"
        )
        if task == "Upload new documents":
            folder_upload = st.file_uploader("Upload a zip file only", type=["zip"], key="folder_upload")
            # old_upload = st.file_uploader("Upload your previous CSV file", type=["csv"], key="old_upload")

        if task == "Type input manually":
            manual_update(old_upload)


    st.subheader('3. Enter the file name for the new CSV file')
    file_name = st.text_input("The new file name:", value='Financial_Diaries_updated.csv')
    if st.button("Submit"):
        final_output = process_data(folder_upload, old_upload, file_name)
        display_data_structure(final_output)
        display_overview(final_output)
        # column_name = st.selectbox(
        #     'Choose between columns',
        #     ('Transaction_Name', 'Transaction_Comment'))
        # n = st.sidebar.slider('Select number of top frequent words:', 1, 30, 10)
        #
        # # Placeholder for the visualization
        # placeholder = st.empty()
        # fig = interactive_transaction_analysis(final_output, column_name, n)
        # placeholder.plotly_chart(fig)


if __name__ == "__main__":
    main()

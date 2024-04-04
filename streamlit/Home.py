import streamlit as st
import os
import csv
import sys
from pathlib import Path
import pandas as pd
from streamlit.components.v1 import components
# # sourcecode_path = Path(os.getcwd()).parent / 'source_code'
# sourcecode_path = Path(os.getcwd()).parent
# sys.path.append(sourcecode_path)
# from word2csv import get_file_locations, extract_info_from_docx, convert_table_to_csv_file
# # from source_code.geo_viz import init_map, create_point_map, plot_from_df, load_df, load_map_region, load_map_state, main_region, main_state, plots
# from data_cleaning import clean_date_format, fix_year_format, clean_mem_status, clean_transaction_amount
# from pdf2csv_easyOCR import ocr_result
from source_code.word2csv import get_file_locations, extract_info_from_docx, convert_table_to_csv_file
# from source_code.geo_viz import init_map, create_point_map, plot_from_df, load_df, load_map_region, load_map_state, main_region, main_state, plots
from source_code.data_cleaning import clean_date_format, fix_year_format, clean_mem_status, clean_transaction_amount
# from source_code.pdf2csv_easyOCR import ocr_result

# st.title('Welcome!')
st.title('Welcome to :blue[Docx2Dashboard]🪄')
st.header('Introducing Our Solution for Digital Transformation', divider='grey')
st.subheader("💁‍♀️ Empowering unbanked Nigerian women through Digital Transformation.")
st.subheader("📑 Convert paper-based financial records into CSV files effortlessly.")
st.subheader("📊 Access dynamic visual dashboards for comprehensive financial management.")

# introduction = '''
# Empowering unbanked Nigerian women through digital transformation.
#
#
# '''
# st.markdown(introduction)
st.text("This App is developed in the GWU Data Science Capstone Project with World Bank Social Sustainability and Inclusion unit")
# st.text("")
st.text("Team Member: Brooklyn Chen, Jiwoo Suh, Sanjana Godolkar")

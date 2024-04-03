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
from source_code.pdf2csv_easyOCR import ocr_result

st.title('Doc 2 :blue[Dashboard]')

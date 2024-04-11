import streamlit as st
import os
import pandas as pd

from utils import *
st.set_page_config(
    layout='wide',
    page_title="Vizualization Dashboard",
    page_icon="📊"
)

st.markdown(
    f'<h1 style="background-color:#0169CA; color:white; text-align:center; border-radius: 5px;">Visualization Dashboard</h1>',
    unsafe_allow_html=True
)

def main():
    st.write("💡Unlock insights with our Visualization Dashboard")
    old_upload = 'Financial_Diaries_final.csv'
    df = pd.read_csv(old_upload)
    st.subheader('Data Structure')
    display_data_structure(df)
    display_overview(df)
    column_name = st.selectbox(
        'Choose between columns',
        ('Transaction_Name', 'Transaction_Comment'))
    n = st.sidebar.slider('Select number of top frequent words:', 1, 30, 10)

    # Placeholder for the visualization
    placeholder = st.empty()
    fig = interactive_transaction_analysis(df, column_name, n)
    placeholder.plotly_chart(fig)


if __name__ == "__main__":
    main()
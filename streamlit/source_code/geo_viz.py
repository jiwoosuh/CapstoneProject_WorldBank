import folium
import geopandas
import pandas as pd
import streamlit as st
from shapely.geometry import Point
from streamlit_folium import st_folium
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def init_map(center=(9.0820, 11.5), zoom_start=6, map_type="cartodbpositron"):
    return folium.Map(location=center, zoom_start=zoom_start, tiles=map_type)

def create_point_map(df, latitude_col, longitude_col):
    # Cleaning
    df[[latitude_col, longitude_col]] = df[[latitude_col, longitude_col]].apply(pd.to_numeric, errors='coerce')
    # Convert PandasDataFrame to GeoDataFrame
    df['coordinates'] = df[[latitude_col, longitude_col]].values.tolist()
    df['coordinates'] = df['coordinates'].apply(Point)
    df = geopandas.GeoDataFrame(df, geometry='coordinates')
    df = df.dropna(subset=[latitude_col, longitude_col, 'coordinates'])
    return df

def plot_from_df(df, folium_map, latitude_col, longitude_col, data_col):
    df = create_point_map(df, latitude_col, longitude_col)
    for i, row in df.iterrows():
        folium.Marker([row[latitude_col], row[longitude_col]],
                      tooltip=f'{row[data_col]}',
                      opacity=row['Transaction_Amount'],
                      icon=folium.Icon(icon='cloud')).add_to(folium_map)
    return folium_map

def load_df():
    df = pd.read_csv('test.csv')
    return df


FACT_BACKGROUND = """
                    <div style="width: 100%;">
                        <div style="
                                    background-color: #ECECEC;
                                    border: 1px solid #ECECEC;
                                    padding: 1.5% 1% 1% 3.5%;
                                    border-radius: 10px;
                                    width: 100%;
                                    color: black;
                                    white-space: nowrap;
                                    ">
                          <p style="font-size:20px; color: black;">{}</p>
                          <p style="font-size:18px; line-height: 0.5; text-indent: 10px;"">{}</p>
                          <p style="font-size:18px; line-height: 0.5; text-indent: 10px;"">{}</p>
                          <p style="font-size:18px; line-height: 0.5; text-indent: 10px;"">{}</p>
                          <p style="font-size:18px; line-height: 0.5; text-indent: 10px;"">{}</p>                          
                        </div>
                    </div>
                    """

def load_map_region():
    # Load the map
    m = init_map()  # init
    df = load_df()  # load data
    m = plot_from_df(df, m, latitude_col='Region_Latitude', longitude_col='Region_Longitude', data_col='Region')
    return m

def load_map_state():
    # Load the map
    m = init_map()  # init
    df = load_df()  # load data
    m = plot_from_df(df, m, latitude_col='State_Latitude', longitude_col='State_Longitude', data_col='State')
    return m

def main_region():
    # format page
    # st.set_page_config(layout='wide')

    # load map data @st.cache_resource
    m = load_map_region()
    # init stored values
    if "selected_id" not in st.session_state:
        st.session_state.selected_id = None

    # main information line: includes map location
    _, r2_col1, r2_col2, r2_col3, _ = st.columns([1, 4.5, 1, 6, 1])
    with r2_col1:
        # info sidebar
        r2_col1.markdown('## Region Information')
        text1, text2, text3, text4, text5 = "Respondent", "Total:", "WAG:", "NON WAG:", ""
        st.markdown(FACT_BACKGROUND.format(text1, text2, text3, text4, text5), unsafe_allow_html=True)
        st.markdown("""<div style="padding-top: 15px"></div>""", unsafe_allow_html=True)
        text1, text2, text3, text4, text5 = "Transaction Amount", "Average:", "Average Income:", "Average Expenditure:", ""
        st.markdown(FACT_BACKGROUND.format(text1, text2, text3, text4, text5), unsafe_allow_html=True)
        st.markdown("""<div style="padding-top: 15px"></div>""", unsafe_allow_html=True)
        text1, text2, text3, text4, text5 = "Average Transaction Amounts", "2021 WAG:", "2023 WAG:", "2021 NON WAG:", "2023 NON WAG:"
        st.markdown(FACT_BACKGROUND.format(text1, text2, text3, text4, text5), unsafe_allow_html=True)
        st.markdown("""<div style="padding-top: 15px"></div>""", unsafe_allow_html=True)
        text1, text2, text3, text4, text5 = "Main Transaction Categories", "A", "B", "C", ""
        st.markdown(FACT_BACKGROUND.format(text1, text2, text3, text4, text5), unsafe_allow_html=True)

        # white space
        for _ in range(10):
            st.markdown("")


    # white space
    with r2_col2:
        st.write("")

    # map container
    with r2_col3:
        level1_map_data = st_folium(m, height=600, width=900)
        st.session_state.selected_id = level1_map_data['last_object_clicked_tooltip']

        if st.session_state.selected_id is not None:
            st.write(f'You Have Selected: {st.session_state.selected_id}')


def main_state():
    # format page
    # st.set_page_config(layout='wide')

    # load map data @st.cache_resource
    m = load_map_state()
    # init stored values
    if "selected_id" not in st.session_state:
        st.session_state.selected_id = None

    # main information line: includes map location
    _, r2_col1, r2_col2, r2_col3, _ = st.columns([1, 6, 0.5, 7, 1])
    with r2_col1:
        # info sidebar
        r2_col1.markdown('## State Information')
        text1, text2, text3, text4, text5 = "Respondent", "Total:", "WAG:", "NON WAG:", ""
        st.markdown(FACT_BACKGROUND.format(text1, text2, text3, text4, text5), unsafe_allow_html=True)
        st.markdown("""<div style="padding-top: 15px"></div>""", unsafe_allow_html=True)
        text1, text2, text3, text4, text5 = "Transaction Amount", "Average:", "Average Income:", "Average Expenditure:", ""
        st.markdown(FACT_BACKGROUND.format(text1, text2, text3, text4, text5), unsafe_allow_html=True)
        st.markdown("""<div style="padding-top: 15px"></div>""", unsafe_allow_html=True)
        text1, text2, text3, text4, text5 = "Average Transaction Amounts", "2021 WAG:", "2023 WAG:", "2021 NON WAG:", "2023 NON WAG:"
        st.markdown(FACT_BACKGROUND.format(text1, text2, text3, text4, text5), unsafe_allow_html=True)
        st.markdown("""<div style="padding-top: 15px"></div>""", unsafe_allow_html=True)
        text1, text2, text3, text4, text5 = "Main Transaction Categories", "A", "B", "C", ""
        st.markdown(FACT_BACKGROUND.format(text1, text2, text3, text4, text5), unsafe_allow_html=True)

        # white space
        for _ in range(10):
            st.markdown("")

    # white space
    with r2_col2:
        st.write("")

    # map container
    with r2_col3:
        level1_map_data = st_folium(m, height=600, width=900)
        st.session_state.selected_id = level1_map_data['last_object_clicked_tooltip']

        if st.session_state.selected_id is not None:
            st.write(f'You Have Selected: {st.session_state.selected_id}')


def plots(geo):
    if st.session_state.selected_id is not None:
        _, r2_col1, r2_col2, r2_col3, _ = st.columns([1, 6, 0.5, 7, 1])
        with r2_col1:
            df = load_df()
            df_filtered = df[df[geo] == st.session_state.selected_id]
            df_filtered['Transaction_Amount_log'] = np.log(df_filtered['Transaction_Amount'])

            # Plot histogram and density plot after log
            mean_val = df_filtered["Transaction_Amount_log"].mean()
            median_val = df_filtered["Transaction_Amount_log"].median()
            mode_val = stats.mode(df_filtered["Transaction_Amount_log"]).mode.item()

            plt.figure(figsize=(6, 4))
            sns.histplot(df_filtered["Transaction_Amount_log"], kde=True, color='skyblue')
            plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=1, label='Mean: {:.2f}'.format(mean_val))
            plt.axvline(median_val, color='green', linestyle='dashed', linewidth=1, label='Median: {:.2f}'.format(median_val))
            plt.axvline(mode_val, color='orange', linestyle='dashed', linewidth=1, label='Mode: {:.2f}'.format(mode_val))
            plt.title('Histogram and Density Plot of {}'.format("log(Transaction Amount)"), fontweight='bold')
            plt.xlabel('log(Transaction Amount)', fontweight='bold')
            plt.ylabel('Frequency / Density', fontweight='bold')
            plt.legend()
            combined_fig = plt.gcf()
            st.pyplot(combined_fig)

            # Word Cloud
            # st.subheader("Word Cloud for Transaction Name")
            custom_stopwords = ["the", "and", "to", "of", "in", "for", "on", "with", "by", "from", "at", "is", "are", "was",
                                "were", "it", "that", "this", "an", "as", "or", "be", "have", "has", "not", "no", "can",
                                "could", "but", "so", "if", "when", "where", "how", "why", "which", "cost", "income",
                                "weekly"]

            transaction_names = df_filtered['Transaction_Name'].str.lower().str.split()
            transaction_names = [[word for word in words if word not in custom_stopwords] for words in transaction_names]
            transaction_names_str = ' '.join([' '.join(words) for words in transaction_names])

            wordcloud = WordCloud(width=600, height=400,
                                  background_color='white',
                                  min_font_size=10).generate(transaction_names_str)
            plt.figure(figsize=(6, 4))
            st.set_option('deprecation.showPyplotGlobalUse', False)
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.title('Word Cloud for Transaction Name', fontweight='bold')
            plt.axis('off')
            st.pyplot()

        # white space
        with r2_col2:
            st.write("")
        # Split violin plot
        # st.subheader("Distribution of log(Transaction Amount) by Member Status")
        # plt.figure(figsize=(10, 6))
        # palette_colors = {'NON WAG': 'lightblue', 'WAG': 'lightpink'}
        # sns.violinplot(x='Transaction_Nature', y='Transaction_Amount_log', hue='Member_Status', data=df_filtered,
        #                split=True,
        #                palette=palette_colors, inner="quart")
        #
        # plt.xlabel('Transaction_Nature', fontweight='bold')
        # plt.ylabel('Transaction Amount (log)', fontweight='bold')
        # plt.title('Split Violin Plot of log(Transaction Amount) by Transaction Nature and Member Status')
        #
        # plt.xticks(rotation=45)
        # plt.legend(title='Member Status', loc='lower left')
        # st.pyplot()

        with r2_col3:
            # Pivot
            # st.subheader("Average Transaction Amount by Member Status")
            pivot_df = df_filtered.pivot_table(index='Transaction_Nature', columns='Member_Status',
                                               values='Transaction_Amount', aggfunc='mean')

            plt.figure(figsize=(6, 4.3))
            sns.heatmap(pivot_df, cmap='Blues', annot=True, fmt=".1f", linewidths=.5)
            plt.title('Average Transaction Amount by Transaction Nature and Member Status', fontweight='bold')
            plt.xlabel('Member Status', fontweight='bold')
            plt.ylabel('Transaction Nature', fontweight='bold')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot()

            # Mirror chart
            # st.subheader("Average Transaction Amounts between 2023 and 2021")
            df_filtered['Formatted_Date'] = pd.to_datetime(df_filtered['Formatted_Date'])

            df_wag = df_filtered[df_filtered['Member_Status'] == 'WAG']
            df_nwag = df_filtered[df_filtered['Member_Status'] == 'NON WAG']

            df_2021_n = df_nwag[df_nwag['Formatted_Date'].dt.year == 2021].groupby('Transaction_Nature')[
                'Transaction_Amount'].mean().reset_index()
            df_2021_w = df_wag[df_wag['Formatted_Date'].dt.year == 2021].groupby('Transaction_Nature')[
                'Transaction_Amount'].mean().reset_index()
            df_2023_n = df_nwag[df_nwag['Formatted_Date'].dt.year == 2023].groupby('Transaction_Nature')[
                'Transaction_Amount'].mean().reset_index()
            df_2023_w = df_wag[df_wag['Formatted_Date'].dt.year == 2023].groupby('Transaction_Nature')[
                'Transaction_Amount'].mean().reset_index()

            fig, ax = plt.subplots(figsize=(6, 4.5))

            bar_width = 0.35
            bar_positions_2021_n = range(len(df_2021_n))
            bar_positions_2021_w = [pos + bar_width for pos in bar_positions_2021_n]
            bar_positions_2023_n = range(len(df_2023_n))
            bar_positions_2023_w = [pos + bar_width for pos in bar_positions_2023_n]

            bars_2021_n = ax.barh(bar_positions_2021_n, -df_2021_n['Transaction_Amount'], bar_width, color='steelblue',
                                  label='2021 NON WAG')
            bars_2021_w = ax.barh(bar_positions_2021_w, -df_2021_w['Transaction_Amount'], bar_width, color='lightgreen',
                                  label='2021 WAG')
            bars_2023_n = ax.barh(bar_positions_2023_n, df_2023_n['Transaction_Amount'], bar_width, color='lightblue',
                                  label='2023 NON WAG')
            bars_2023_w = ax.barh(bar_positions_2023_w, df_2023_w['Transaction_Amount'], bar_width, color='green',
                                  label='2023 WAG')

            ax.set_yticks([pos + bar_width / 2 for pos in bar_positions_2021_n])
            ax.set_yticklabels(df_2021_n['Transaction_Nature'])

            ax.set_xlabel('Average Transaction Amount', fontweight='bold')
            ax.set_ylabel('Transaction Nature', fontweight='bold')

            for bars in [bars_2021_n, bars_2021_w, bars_2023_n, bars_2023_w]:
                for bar in bars:
                    value = bar.get_width()
                    ha = 'right' if value < 0 else 'left'
                    ax.text(value, bar.get_y() + bar.get_height() / 2,
                            f'{abs(value):.2f}', ha=ha, va='center', color='black')

            ax.set_xticklabels([f'{abs(x)}' for x in ax.get_xticks()])
            ax.legend(loc='center right')
            plt.title('Average Transaction Amounts by Transaction Nature and Member Status', fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)


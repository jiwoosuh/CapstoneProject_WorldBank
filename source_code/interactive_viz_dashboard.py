import streamlit as st
import os
import pandas as pd
from dataprep.eda import create_report
import nltk
from nltk.corpus import stopwords
from collections import Counter
import matplotlib.pyplot as plt

cwd = os.getcwd()
df = pd.read_csv('../Financial_Diaries_Modified.csv')
# report = create_report(df)

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
def preprocess_text(column):
    tokens = []
    column = column.dropna()
    for sent in column:  # Drop NaN rows
        words = nltk.word_tokenize(sent.lower())
        filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
        tokens.extend(filtered_words)
    return tokens

def get_frequent_words(column, n=10):
    all_words = preprocess_text(column)
    word_counts = Counter(all_words)
    frequent_words = word_counts.most_common(n)
    return frequent_words


frequent_words = get_frequent_words(df['Transaction_Comment'])

print("Frequent words:")
for word, count in frequent_words:
    print(f"{word}: {count}")

words = [pair[0] for pair in frequent_words]
frequencies = [pair[1] for pair in frequent_words]

# Plotting
plt.figure(figsize=(10, 6))
plt.bar(words, frequencies, color='skyblue')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.title('Top 10 Frequent Words')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Streamlit app
st.title('Top n Frequent Words in Transaction Comments')

# Sidebar to adjust n
n = st.sidebar.slider('Select number of top frequent words:', 1, 30, 10)

# Get frequent words from a specific column
column_name = st.selectbox(
    'Choose between columns',
    ('Transaction_Category1', 'Transaction_Name', 'Transaction_Comment'))  # Change this to the name of your desired column
frequent_words = get_frequent_words(df[column_name], n)

# Extract words and their frequencies
words = [pair[0] for pair in frequent_words]
frequencies = [pair[1] for pair in frequent_words]

# Plotting
st.write("### Visualization:")
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(words, frequencies, color='skyblue')
ax.set_xlabel('Words')
ax.set_ylabel('Frequency')
ax.set_title(f'Top {n} Frequent Words')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Display plot using Streamlit
st.pyplot(fig)

st.write("### Frequent Words:")
data = {'Word': [pair[0] for pair in frequent_words],
        'Count': [pair[1] for pair in frequent_words]}
df_frequent_words = pd.DataFrame(data)
st.table(df_frequent_words)
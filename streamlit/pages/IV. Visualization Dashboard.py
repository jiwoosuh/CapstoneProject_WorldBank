import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

####################################################################################################################################
#This code has a bunch of potential plots for interactive dashboard. We will discuss in person about which ones to keep and which
#ones to discard. Please Ignore for now. 
#Still need to add the Left side bar and intereactive features from plotly.
####################################################################################################################################


# Load the dataset
df = pd.read_csv('Financial_Diaries_Modified.csv')

# Setting a light blue color palette for seaborn
light_blue_palette = sns.light_palette("skyblue", reverse=False)
darker_blue_palette = sns.color_palette("Blues")
# Setting a blue color palette for seaborn
blue_palette = sns.light_palette("skyblue", reverse=True)

# Streamlit layout starts here
st.set_page_config(layout="wide")  # Set the page to wide mode

# Create a figure to hold the first five subplots with a 2x3 layout (we will adjust later)
fig, axes = plt.subplots(2, 3, figsize=(20, 10))  # Adjust the size as needed

# Flatten the axes array for easy iterating
axes = axes.flatten()

# Count for Member Status for the pie chart
member_status_counts = df['Member_Status'].value_counts()

# 1. Bar chart for FD_Name counts(***)
sns.countplot(y='FD_Name', data=df, ax=axes[0], palette=darker_blue_palette)
axes[0].set_title('FD_Name Count')

# # 2. Pie chart for Member Status distribution with shades of blue
# pie_chart_palette = sns.light_palette("skyblue", reverse=False, n_colors=len(member_status_counts))
# axes[1].pie(
#     member_status_counts, 
#     labels=member_status_counts.index, 
#     autopct='%1.1f%%', 
#     colors=pie_chart_palette  # Use the shades of blue here
# )
# axes[1].set_title('Member Status Distribution')
# # 3. Histogram for Transaction Amounts with zoom into the first few bars
# sns.histplot(df['Transaction_Amount'], bins=30, kde=True, ax=axes[2], color=darker_blue_palette[4])
# axes[2].set_title('Transaction Amount Distribution')
# axes[2].set_xlim(0, df['Transaction_Amount'].quantile(0.95))  # Zoom into 95% quantile

# 4. Bar chart for Transaction Nature counts(***)
sns.countplot(x='Transaction_Nature', data=df, ax=axes[3], palette=darker_blue_palette)
axes[3].set_title('Transaction Nature Count')

# 5. Stacked bar chart for Transaction_Type by Transaction_Category1 with adjusted colors(***)
transaction_type_category_counts = df.groupby(['Transaction_Category1', 'Transaction_Type']).size().unstack().fillna(0)
transaction_type_category_counts.plot(kind='bar', stacked=True, colormap=sns.light_palette("navy", as_cmap=True), ax=axes[4])
axes[4].set_title('Transaction Type by Transaction Category1')

# Removing the 6th subplot (we'll keep this space empty, or you could add another plot here)
fig.delaxes(axes[5])

plt.tight_layout()
st.pyplot(fig)

# Now let's create the State and Region bar plot with stacked bars and labels
# Prepare the data for the stacked bar chart
grouped = df.groupby(['State', 'Region']).size().rename('Count').reset_index()
pivot_df = grouped.pivot(index='State', columns='Region', values='Count').fillna(0)

# Plot the stacked bars
fig, ax = plt.subplots(figsize=(20, 5))

# Create a color palette for blue shades
palette = sns.color_palette("Blues", len(pivot_df.columns))

# Store the bottom of each bar to stack them
bottom = np.zeros(len(pivot_df))

# Iterate through the DataFrame to stack bars and add labels
for i, (region, color) in enumerate(zip(pivot_df.columns, palette)):
    # Stack the bar for the current region
    bars = ax.bar(pivot_df.index, pivot_df[region], bottom=bottom, color=color, label=region)
    # Update the bottom positions for the next stack
    bottom += pivot_df[region].values

# Add labels horizontally inside the bars
for state_index, state in enumerate(pivot_df.index):
    cum_height = 0
    for region_index, (region, count) in enumerate(pivot_df.loc[state].items()):
        if count > 0:  # Only add a label if there's a count for the region
            ax.text(
                state_index, 
                cum_height + count / 2, 
                region, 
                ha='center', 
                va='center', 
                color='black', 
                fontsize=9
            )
            cum_height += count

# Customize the plot
ax.set_ylabel('Counts')
ax.set_title('Count of Unique Regions within Each State')
ax.set_xticklabels(pivot_df.index, rotation=45, ha='right')

plt.tight_layout()

# Display in Streamlit
st.pyplot(fig)


# Set page config
st.set_page_config(page_title="Financial Diaries Dashboard", layout="wide")

# Title for the dashboard
st.title("Financial Diaries Dashboard")

# State vs Transaction Type vs Amounts
st.subheader('Total Transaction Amounts by State and Transaction Type')
plt.figure(figsize=(15, 8))
sns.barplot(x='State', y='Transaction_Amount', hue='Transaction_Type', data=df, estimator=sum, palette='Blues')
plt.title('Total Transaction Amounts by State and Transaction Type')
plt.xlabel('State')
plt.ylabel('Sum of Transaction Amounts')
plt.legend(title='Transaction Type')
st.pyplot(plt.gcf())

# Region vs Transaction Type vs Amounts
st.subheader('Total Transaction Amounts by Region and Transaction Type')
plt.figure(figsize=(15, 8))
sns.barplot(x='Region', y='Transaction_Amount', hue='Transaction_Type', data=df, estimator=sum, palette='Blues')
plt.title('Total Transaction Amounts by Region and Transaction Type')
plt.xlabel('Region')
plt.ylabel('Sum of Transaction Amounts')
plt.legend(title='Transaction Type')
st.pyplot(plt.gcf())

# Week vs Transaction Category
st.subheader('Transaction Categories per Week')
plt.figure(figsize=(15, 8))
sns.countplot(x='Week', hue='Transaction_Category1', data=df)
plt.title('Transaction Categories per Week')
plt.xlabel('Week')
plt.ylabel('Count')
plt.legend(title='Transaction Category', bbox_to_anchor=(1.05, 1), loc=2)
st.pyplot(plt.gcf())

# Week vs Transaction Type vs Amounts
st.subheader('Total Transaction Amounts by Week and Transaction Type')
plt.figure(figsize=(15, 8))
sns.barplot(x='Week', y='Transaction_Amount', hue='Transaction_Type', data=df, estimator=sum, palette='Blues')
plt.title('Total Transaction Amounts by Week and Transaction Type')
plt.xlabel('Week')
plt.ylabel('Sum of Transaction Amounts')
plt.legend(title='Transaction Type')
st.pyplot(plt.gcf())

# Category vs Amounts
st.subheader('Total Transaction Amounts by Category')
plt.figure(figsize=(15, 8))
sns.barplot(x='Transaction_Category1', y='Transaction_Amount', data=df, estimator=sum, palette='Blues')
plt.title('Total Transaction Amounts by Category')
plt.xlabel('Transaction Category')
plt.xticks(rotation=90)
plt.ylabel('Sum of Transaction Amounts')
st.pyplot(plt.gcf())

# State-wise Average Transaction Amount over Time
st.subheader('State-wise Average Transaction Amount Over Time')
plt.figure(figsize=(14, 7))
sns.lineplot(data=df, x='Week', y='Transaction_Amount', hue='State', estimator='mean', marker='o', palette='Blues')
plt.title('State-wise Average Transaction Amount Over Time')
plt.xlabel('Week')
plt.ylabel('Average Transaction Amount')
plt.legend(title='State')
st.pyplot(plt.gcf())

# Transaction Type Frequency by Region
st.subheader('Transaction Type Frequency by Region')
transaction_type_freq = df.groupby(['Region', 'Transaction_Type']).size().unstack(fill_value=0)
transaction_type_freq.plot(kind='bar', stacked=True, figsize=(14, 7), color=sns.color_palette('Blues'))
plt.title('Transaction Type Frequency by Region')
plt.xlabel('Region')
plt.ylabel('Frequency')
st.pyplot(plt.gcf())

# Transaction Category vs. Member Status vs. Amount (Bubble Chart)
st.subheader('Transaction Category vs. Member Status vs. Amount')
plt.figure(figsize=(14, 7))
bubble_size = df['Transaction_Amount'].div(df['Transaction_Amount'].max()) * 1000  # Normalizing for bubble size
sns.scatterplot(data=df, x='Transaction_Category1', y='Member_Status', size=bubble_size, legend=False, sizes=(20, 500), hue='Transaction_Type', palette='Blues')
plt.title('Transaction Category vs. Member Status vs. Amount')
plt.xlabel('Transaction Category')
plt.ylabel('Member Status')
st.pyplot(plt.gcf())

# Boxplot of Transaction Amounts by Category and Type
st.subheader('Boxplot of Transaction Amounts by Category and Type')
plt.figure(figsize=(14, 7))
sns.boxplot(data=df, x='Transaction_Category1', y='Transaction_Amount', hue='Transaction_Type', palette='Blues')
plt.title('Boxplot of Transaction Amounts by Category and Type')
plt.xlabel('Transaction Category')
plt.ylabel('Transaction Amount')
plt.legend(title='Transaction Type')
plt.xticks(rotation=90)
st.pyplot(plt.gcf())

# Expenditure vs. Income over Time
st.subheader('Expenditure vs. Income Over Time')
df['Transaction_Type_Coded'] = df['Transaction_Type'].map({'Income': 1, 'Expenditure': -1})
df['Signed_Amount'] = df['Transaction_Amount'] * df['Transaction_Type_Coded']
plt.figure(figsize=(14, 7))
df_income = df[df['Transaction_Type_Coded'] == 1].groupby('Week')['Transaction_Amount'].sum()
df_expenditure = df[df['Transaction_Type_Coded'] == -1].groupby('Week')['Transaction_Amount'].sum().abs()
plt.plot(df_income.index, df_income.values, label='Income', marker='o', color='blue')
plt.plot(df_expenditure.index, df_expenditure.values, label='Expenditure', marker='o', color='darkblue')
plt.title('Expenditure vs. Income Over Time')
plt.xlabel('Week')
plt.ylabel('Amount')
plt.legend()
st.pyplot(plt.gcf())

# Comparative Histogram of Transaction Amounts
st.subheader('Comparative Histogram of Transaction Amounts')
plt.figure(figsize=(14, 7))
sns.histplot(data=df, x='Transaction_Amount', hue='State', element='step', bins=30, palette='Blues')
plt.title('Comparative Histogram of Transaction Amounts')
plt.xlabel('Transaction Amount')
plt.ylabel('Density')
plt.legend(title='State')
st.pyplot(plt.gcf())

# Scatter Plot of Transaction Amount vs. Week with Hue for Region
st.subheader('Scatter Plot of Transaction Amount vs. Week with Hue for Region')
plt.figure(figsize=(14, 7))
sns.scatterplot(data=df, x='Week', y='Transaction_Amount', hue='Region', palette='Blues')
plt.title('Scatter Plot of Transaction Amount vs. Week with Hue for Region')
plt.xlabel('Week')
plt.ylabel('Transaction Amount')
plt.legend(title='Region')
st.pyplot(plt.gcf())

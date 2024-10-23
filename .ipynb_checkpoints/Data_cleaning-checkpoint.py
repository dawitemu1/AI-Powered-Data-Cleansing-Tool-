import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno  # Import the missingno library

# Title of the app
st.title('Data Quality and Data cleaning')

# Upload file from local machine
uploaded_file = st.file_uploader("Upload your file", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    # Check the file type and read the file accordingly
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(uploaded_file)
    
    # Display the dataset
    st.write("Uploaded Dataset:")
    st.write(df)
    
    # Button to show total missing values
    if st.button("Show Total Missing Values"):
        null_values = df.isnull().sum()
        st.write("Total Missing Values:\n")
        st.table(null_values)

    # Button to show percentage of missing values
    if st.button("Show Percentage of Missing Values"):
        percentage_missing = (df.isnull().sum() / len(df)) * 100
        st.write("Percentage of Missing Values:\n")
        st.table(percentage_missing)
    
    # Button to visualize missing values as a heatmap
    if st.button("Show Missing Values Heatmap"):
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(df.isnull(), cbar=False, cmap='viridis', ax=ax)
        plt.title('Missing Values Heatmap')
        st.pyplot(fig)
    
    # Button to visualize total missing values as a bar chart
    if st.button("Show Missing Values Bar Chart"):
        null_values = df.isnull().sum()
        fig, ax = plt.subplots(figsize=(12, 6))
        null_values.plot(kind='bar', color='orange', ax=ax)
        plt.title('Total Missing Values by Column')
        plt.ylabel('Number of Missing Values')
        st.pyplot(fig)
    
    # Button to visualize missing values using missingno library matrix
    if st.button("Show Missingno Matrix"):
        fig = msno.matrix(df)
        st.pyplot(fig.figure)
    
    # Button to visualize missing values using missingno bar chart
    if st.button("Show Missingno Bar Chart"):
        fig = msno.bar(df)
        st.pyplot(fig.figure)
    
    # Button to visualize missing values using missingno heatmap
    if st.button("Show Missingno Heatmap"):
        fig = msno.heatmap(df)
        st.pyplot(fig.figure)

    # Dropdown to select a column for distinct value check
    selected_column = st.selectbox("Select a Column to Check Distinct Values", df.columns)
    
    # Button to show distinct values for the selected column
    if st.button("Show Distinct Values for Selected Column"):
        distinct_values = df[selected_column].unique()
        st.write(f"Distinct values in '{selected_column}':")
        st.write(distinct_values)

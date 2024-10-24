import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr
from io import BytesIO
import calendar  # To get month names
from io import StringIO

# Configuring the page with Title, icon, and layout
st.set_page_config(
    page_title="EDA for Loan status",
    page_icon="/home/U5MRIcon.png",
    layout="wide",
    menu_items={
        'Get Help': 'https://helppage.ethipiau5m.com',
        'Report a Bug': 'https://bugreport.ethipiau5m.com',
        'About': 'https://ethiopiau5m.com',
    },
)

# Custom CSS to adjust spacing
custom_css = """
<style>
    div.stApp {
        margin-top: -90px !important;  /* We can adjust this value as needed */
    }
</style>
"""
# st.markdown(custom_css, unsafe_allow_html=True)
# st.image("logo.jpg", width=400)  # Change "logo.png" to the path of your logo image file
# # Setting the title with Markdown and center-aligning
st.markdown('<h1 style="text-align: center;">AI Powered Data Quality Tool </h1>', unsafe_allow_html=True)

# Defining background color
st.markdown(
    """
    <style>
    body {
        background-color: #f5f5f5;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Defining header color and font
st.markdown(
    """
    <style>
    h1 {
        color: #800080;  /* Blue color */
        font-family: 'Helvetica', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)


@st.cache_data
def load_data(file_path):
    return pd.read_excel(file_path)




# Load cleaned data
def load_data(file):
    if isinstance(file, str):  # If the input is a string path
        if file.endswith('.xlsx') or file.endswith('.xls'):
            return pd.read_excel(file)
        elif file.endswith('.csv'):
            return pd.read_csv(file)
        else:
            st.error("Unsupported file format.")
            return None
    elif isinstance(file, st.runtime.uploaded_file_manager.UploadedFile):  # If the input is an UploadedFile object
        file_extension = file.name.split('.')[-1]
        if file_extension in ['xlsx', 'xls']:
            return pd.read_excel(BytesIO(file.getvalue()))
        elif file_extension == 'csv':
            return pd.read_csv(BytesIO(file.getvalue()))
        else:
            st.error("Unsupported file format.")
            return None
    else:
        st.error("Invalid file input.")
        return None

# Default path for cleaned data
default_cleaned_data_path = 'D:/CBE_project_related/LOAN WITH COLLATERAL/cleaned_Ex_data_2014-2024.xlsx'

# File uploader for user input
uploaded_file = st.file_uploader("Upload your Excel or CSV file", type=["xlsx", "xls", "csv"])

if uploaded_file is not None:
    # Load the uploaded file
    dataset = load_data(uploaded_file)
else:
    # Load the default file
    dataset = load_data(default_cleaned_data_path)

# Display the dataset
# if dataset is not None:
#     st.write(dataset)
# else:
#     st.write("No data available.")

# List of features to exclude from the dropdown menu
excluded_features = ['year_month']

# Filter out excluded features
allowed_features = [col for col in dataset.columns if col not in excluded_features]

# Function to create a horizontal line with custom styling
def horizontal_line(height=1, color="blue", margin="0.5em 0"): # color="#ddd"
    return f'<hr style="height: {height}px; margin: {margin}; background-color: {color};">'




####################### 3. Data quality ###################################################
import pandas as pd
import streamlit as st
import missingno as msno
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from scipy.stats import zscore
import numpy as np
import seaborn as sns

# Function to fill missing values
def fill_missing_values(dataset):
    # Fill categorical features using mode
    categorical_cols = dataset.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        dataset[col].fillna(dataset[col].mode()[0], inplace=True)

    # Fill numerical features using KNN
    numerical_cols = dataset.select_dtypes(include=['number']).columns
    if len(numerical_cols) > 0:
        imputer = KNNImputer(n_neighbors=5)
        dataset[numerical_cols] = imputer.fit_transform(dataset[numerical_cols])

    return dataset

# Function to create a downloadable CSV
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# Function to display 'Null' in red where there are missing values
def highlight_missing(val):
    if pd.isnull(val):
        return 'background-color: red; color: white; text-align: center;', 'Null'  # Set red background and white text for 'Null'
    else:
        return '', val  # No styling for non-missing values


# Set the Pandas Styler max_elements option to allow rendering of larger dataframes
pd.set_option("styler.render.max_elements", 5001720)  # Adjust this based on the dataset size

# Function to apply red color styling to 'Null' values
def highlight_missing(val):
    if val == 'Null':  # Check if the cell value is 'Null'
        return 'color: red;'  # Apply red color to 'Null' text
    else:
        return ''  # No style for other values

# Modified display for 'Each Missed Value by Row and Column'
@st.cache_data
def display_missing_values(dataset, view_option):
    if view_option == 'Total Null Values':
        total_nulls = dataset.isnull().sum().sum()
        st.write(f'Total Null Values in the dataset: {total_nulls}')
        
    elif view_option == 'Total Nulls by Table':
        nulls_by_column = dataset.isnull().sum()
        nulls_table = pd.DataFrame(nulls_by_column, columns=['Total Nulls'])
        nulls_table['Percentage'] = (nulls_table['Total Nulls'] / len(dataset)) * 100
        st.write('Total Nulls by Column:')
        st.dataframe(nulls_table[nulls_table['Total Nulls'] > 0])  # Display only columns with nulls

    elif view_option == 'Total Nulls by Percentage':
        nulls_percentage = (dataset.isnull().sum() / len(dataset)) * 100
        nulls_percentage_df = pd.DataFrame(nulls_percentage, columns=['Percentage Nulls'])
        st.write('Percentage of Nulls by Column:')
        st.dataframe(nulls_percentage_df)  # Show all columns with their percentages

    elif view_option == 'Bar Graph of Missing Values':
        st.write('Bar Graph of Missing Values by Feature:')
        fig, ax = plt.subplots(figsize=(10, 6))  # Create a new figure
        msno.bar(dataset, ax=ax)  # Pass the axis to msno.bar()
        st.pyplot(fig)

    elif view_option == 'Total Nulls by Row':
        nulls_by_row = dataset.isnull().sum(axis=1)
        st.write('Total Nulls by Row:')
        st.dataframe(nulls_by_row[nulls_by_row > 0])  # Display only rows with nulls

    elif view_option == 'Each Missed Value by Row and Column':
        st.write('Missing Values by Row and Column (Red Highlight for Nulls):')
        styled_data = dataset.copy()  # Work with a copy to avoid modifying the original data
        styled_data = styled_data.applymap(lambda x: 'Null' if pd.isnull(x) else x)  # Replace NaN with 'Null'
        styled_data = styled_data.style.applymap(highlight_missing)  # Apply the style for 'Null'
        st.dataframe(styled_data)



# Function to check for duplicates
def check_duplicates(dataset):
    duplicates_count = dataset.duplicated().sum()
    return duplicates_count

# Function to drop duplicates
def drop_duplicates(dataset):
    return dataset.drop_duplicates()

# Function to count unique values for categorical columns
def count_uniqueness(dataset, selected_column):
    unique_counts = dataset[selected_column].value_counts()
    return unique_counts

# Function to create a horizontal line with custom styling
def horizontal_line(height=1, color="blue", margin="0.5em 0"):
    return f'<hr style="height: {height}px; margin: {margin}; background-color: {color};">'

# Function to detect outliers using Z-score or IQR method (without removal)
def detect_outliers(dataset, method, selected_column):
    outliers_detected = None
    
    if method == 'Z-score':
        z_scores = zscore(dataset[selected_column])
        abs_z_scores = np.abs(z_scores)
        outliers_detected = abs_z_scores > 3  # Z-score threshold = 3

    elif method == 'IQR':
        Q1 = dataset[selected_column].quantile(0.25)
        Q3 = dataset[selected_column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers_detected = (dataset[selected_column] < lower_bound) | (dataset[selected_column] > upper_bound)

    return outliers_detected

# Function to remove outliers using Z-score or IQR method
def remove_outliers(dataset, outliers_detected):
    return dataset[~outliers_detected]

# Load your dataset
# Assume dataset is loaded here

# Initialize outliers_detected as None at the top
outliers_detected = None

# Data Quality Section in Sidebar
st.sidebar.header('Data Quality Metrix')
# Separator for Data Overview section
st.sidebar.markdown(horizontal_line(), unsafe_allow_html=True)

# Section for Data Overview (Datatype, Mean, Median, Standard Deviation)
st.sidebar.header('0. Data Profile')

# Function to convert columns to datetime if they represent dates or times
def convert_to_datetime(dataset):
    for col in dataset.columns:
        # Attempt to convert the column to datetime
        temp = pd.to_datetime(dataset[col], errors='coerce')
        # Check if conversion resulted in all NaT or if the original column has valid dates
        if temp.notna().any() and dataset[col].dtype == 'object':
            dataset[col] = temp  # Only update if it's a date/time column
    return dataset

# Conver date columns before displaying the data profile
dataset = convert_to_datetime(dataset)

# Button to show data types, mean, median, standard deviation, min, max, and range
if st.sidebar.button('Data Profile'):
    # Calculate range (max - min)
    range_values = dataset.max(numeric_only=True) - dataset.min(numeric_only=True)

    # Create DataFrame for Data Overview
    data_info = pd.DataFrame({
        'Datatype': dataset.dtypes,
        'Mean': dataset.mean(numeric_only=True),
        'Median': dataset.median(numeric_only=True),
        'Std Deviation': dataset.std(numeric_only=True),
        'Min': dataset.min(numeric_only=True),
        'Max': dataset.max(numeric_only=True),
        'Range': range_values
    })
    
    # Display the data overview table
    st.write("Data Profile (Datatype, Mean, Median, Std Deviation, Min, Max, Range):")
    st.dataframe(data_info)

# Separator before "1. Missing Value"
st.sidebar.markdown(horizontal_line(), unsafe_allow_html=True)

# Section for Missing Values
st.sidebar.header('1. Missing Value')

# Allow the user to choose the method to view missing values
view_option = st.sidebar.radio('Select View for Missing Values', 
                                ['Total Null Values', 
                                 'Total Nulls by Table', 
                                 'Total Nulls by Percentage', 
                                 'Bar Graph of Missing Values', 
                                 'Total Nulls by Row', 
                                 'Each Missed Value by Row and Column'])

# Button to display missing values based on selected option
if st.sidebar.button('Show Missing Values'):
    display_missing_values(dataset, view_option)

# Separator for Fill Missing Values section
st.sidebar.markdown(horizontal_line(), unsafe_allow_html=True)

# New Section for Filling Missing Values
st.sidebar.header('2. Fill Missing Value')
if st.sidebar.button('Fill Missing Values'):
    original_shape = dataset.shape
    dataset = fill_missing_values(dataset)
    st.write(f"Missing values filled. Original shape: {original_shape}, New shape: {dataset.shape}")

    # Allow the user to download the processed data
    csv_data = convert_df_to_csv(dataset)
    st.download_button(
        label="Download Processed Data as CSV",
        data=csv_data,
        file_name='processed_data.csv',
        mime='text/csv'
    )

# Separator for Duplicates section
st.sidebar.markdown(horizontal_line(), unsafe_allow_html=True)

# Section to Check for Duplicates
st.sidebar.header('3. Duplicate Value')

# Button to check for duplicate values
if st.sidebar.button('Check for Duplicates'):
    duplicate_count = check_duplicates(dataset)
    st.write(f"Total Duplicated Rows: {duplicate_count}")

# Option to remove duplicates
if st.sidebar.button('Drop Duplicates'):
    dataset = drop_duplicates(dataset)
    st.write("Duplicate values have been removed.")

    # Allow the user to download the dataset without duplicates
    csv_data = convert_df_to_csv(dataset)
    st.download_button(
        label="Download Data without Duplicates as CSV",
        data=csv_data,
        file_name='cleaned_data_no_duplicates.csv',
        mime='text/csv'
    )

# Separator for Uniqueness count section
st.sidebar.markdown(horizontal_line(), unsafe_allow_html=True)

# Section for Counting Unique Values in Categorical Features
st.sidebar.header('4. Unique Value Counts')

# Dropdown to select categorical columns
categorical_columns = dataset.select_dtypes(include=['object']).columns
selected_column = st.sidebar.selectbox('Select Categorical Column to View Unique Values', categorical_columns)

# Button to show unique values count for the selected column
if st.sidebar.button('Show Unique Values Count'):
    if selected_column:
        unique_counts = count_uniqueness(dataset, selected_column)
        st.write(f"Unique value counts for '{selected_column}':")
        st.dataframe(unique_counts)

# Separator for Outlier detection section
st.sidebar.markdown(horizontal_line(), unsafe_allow_html=True)

# Section to Detect Outliers
st.sidebar.header('5. Outlier Detection')

# Dropdown to select numerical columns for outlier detection
numerical_columns = dataset.select_dtypes(include=['number']).columns
outlier_column = st.sidebar.selectbox('Select Numerical Column for Outlier Detection', numerical_columns)

# Dropdown to select the method for outlier detection (radio box)
outlier_method = st.sidebar.radio('Select Outlier Detection Method', ['Z-score', 'IQR'])

# Button to detect outliers
if st.sidebar.button('Detect Outliers'):
    if outlier_column:
        outliers_detected = detect_outliers(dataset, outlier_method, outlier_column)
        st.write(f"Outliers detected in '{outlier_column}' using {outlier_method}: {outliers_detected.sum()} outliers.")
        
        # Display boxplot before outlier removal
        st.write(f"Box Plot for '{outlier_column}' before outlier removal:")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(x=dataset[outlier_column], ax=ax)
        st.pyplot(fig)

# Separator for Outlier removal section
st.sidebar.markdown(horizontal_line(), unsafe_allow_html=True)

# Sidebar settings for Streamlit
st.sidebar.header('6. Outlier Removal')

# Dropdown to select numerical columns for outlier detection
numerical_columns = dataset.select_dtypes(include=np.number).columns.tolist()
outlier_column = st.sidebar.selectbox('Select Numerical Column for Outlier Removal', numerical_columns)

# Add a radio box for the method of outlier removal
outlier_method_removal = st.sidebar.radio('Select Outlier Removal Method', ['Z-score', 'IQR'])

# Initialize outliers_detected as None
outliers_detected = None

# Button to remove outliers
if st.sidebar.button('Remove Outliers'):
    # Detect outliers first
    if outlier_column:
        outliers_detected = detect_outliers(dataset, outlier_method_removal, outlier_column)
        
        if outliers_detected is not None:
            dataset_cleaned = remove_outliers(dataset, outliers_detected)
            st.write(f"Outliers removed. New dataset shape: {dataset_cleaned.shape}")

            # Display boxplot after outlier removal
            st.write(f"Box Plot for '{outlier_column}' after outlier removal:")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(x=dataset_cleaned[outlier_column], ax=ax)
            st.pyplot(fig)

            # Allow the user to download the dataset without outliers
            csv_data = convert_df_to_csv(dataset_cleaned)
            st.download_button(
                label="Download Data without Outliers as CSV",
                data=csv_data,
                file_name='cleaned_data_no_outliers.csv',
                mime='text/csv'
            )
        else:
            st.write("No outliers detected.")
# Final separator
st.sidebar.markdown(horizontal_line(), unsafe_allow_html=True)

####################### End of Code ############################



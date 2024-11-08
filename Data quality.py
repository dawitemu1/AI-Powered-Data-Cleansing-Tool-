import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr
from io import BytesIO
import calendar  # To get month names
from io import StringIO
from sklearn.impute import KNNImputer
from datetime import datetime
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
st.markdown('<h1 style="text-align: center;">AI Powered Data Cleansing Tool </h1>', unsafe_allow_html=True)

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



# Function to fill missing values with various options
def fill_missing_values(dataset, fill_options):
    # Fill categorical features
    categorical_cols = dataset.select_dtypes(include=['object']).columns
    if fill_options.get("all_categorical"):
        method = fill_options["all_categorical"]
        for col in categorical_cols:
            if method == 'mode':
                dataset[col].fillna(dataset[col].mode()[0], inplace=True)
            elif method == 'constant':
                constant_val = fill_options.get("constant_value_categorical", "Unknown")
                dataset[col].fillna(constant_val, inplace=True)
            elif method == 'ffill':
                dataset[col].fillna(method='ffill', inplace=True)
            elif method == 'bfill':
                dataset[col].fillna(method='bfill', inplace=True)
    else:
        for col in fill_options.get("selected_categorical", []):
            method = fill_options.get("selected_method_categorical")
            if method == 'mode':
                dataset[col].fillna(dataset[col].mode()[0], inplace=True)
            elif method == 'constant':
                constant_val = fill_options.get("constant_value_selected_categorical", "Unknown")
                dataset[col].fillna(constant_val, inplace=True)
            elif method == 'ffill':
                dataset[col].fillna(method='ffill', inplace=True)
            elif method == 'bfill':
                dataset[col].fillna(method='bfill', inplace=True)

    # Fill numerical features
    numerical_cols = dataset.select_dtypes(include=['number']).columns
    if fill_options.get("all_numerical"):
        if fill_options["all_numerical"] == 'KNN':
            imputer = KNNImputer(n_neighbors=5)
            dataset[numerical_cols] = imputer.fit_transform(dataset[numerical_cols])
        else:
            method = fill_options["all_numerical"]
            for col in numerical_cols:
                if method == 'mean':
                    dataset[col].fillna(dataset[col].mean(), inplace=True)
                elif method == 'median':
                    dataset[col].fillna(dataset[col].median(), inplace=True)
                elif method == 'mode':
                    dataset[col].fillna(dataset[col].mode()[0], inplace=True)
                elif method == 'constant':
                    constant_val = fill_options.get("constant_value_numerical", 0.0)
                    dataset[col].fillna(constant_val, inplace=True)
                elif method == 'ffill':
                    dataset[col].fillna(method='ffill', inplace=True)
                elif method == 'bfill':
                    dataset[col].fillna(method='bfill', inplace=True)
    else:
        for col in fill_options.get("selected_numerical", []):
            method = fill_options.get("selected_method_numerical")
            if method == 'mean':
                dataset[col].fillna(dataset[col].mean(), inplace=True)
            elif method == 'median':
                dataset[col].fillna(dataset[col].median(), inplace=True)
            elif method == 'mode':
                dataset[col].fillna(dataset[col].mode()[0], inplace=True)
            elif method == 'constant':
                constant_val = fill_options.get("constant_value_selected_numerical", 0.0)
                dataset[col].fillna(constant_val, inplace=True)
            elif method == 'ffill':
                dataset[col].fillna(method='ffill', inplace=True)
            elif method == 'bfill':
                dataset[col].fillna(method='bfill', inplace=True)

    return dataset

# Function to create a downloadable CSV
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')


    
    
# Set the Pandas Styler max_elements option to allow rendering of larger dataframes
pd.set_option("styler.render.max_elements", 5001720)  # Adjust this based on the dataset size

# Function to apply red color styling to 'Null' values
def highlight_missing(val):
    return 'color: red;' if val == 'Null' else ''

# Function to display missing values based on the selected option
@st.cache_data
def display_missing_values(dataset, selected_features, view_option):
    selected_data = dataset[selected_features]  # Filter data to include only selected features
    
    if view_option == 'Total Null Values':
        total_nulls = selected_data.isnull().sum().sum()
        st.write(f'Total Null Values in the selected features: {total_nulls}')
        
    elif view_option == 'Total Nulls by Table':
        nulls_by_column = selected_data.isnull().sum()
        nulls_table = pd.DataFrame(nulls_by_column, columns=['Total Nulls'])
        nulls_table['Percentage'] = (nulls_table['Total Nulls'] / len(selected_data)) * 100
        st.write('Total Nulls by Column:')
        st.dataframe(nulls_table[nulls_table['Total Nulls'] > 0])  # Display only columns with nulls

    elif view_option == 'Total Nulls by Percentage':
        nulls_percentage = (selected_data.isnull().sum() / len(selected_data)) * 100
        nulls_percentage_df = pd.DataFrame(nulls_percentage, columns=['Percentage Nulls'])
        st.write('Percentage of Nulls by Column:')
        st.dataframe(nulls_percentage_df)  # Show all columns with their percentages

    elif view_option == 'Bar Graph of Missing Values':
        st.write('Bar Graph of Missing Values by Feature:')
        fig, ax = plt.subplots(figsize=(10, 6))
        msno.bar(selected_data, ax=ax)
        st.pyplot(fig)

    elif view_option == 'Total Nulls by Row':
        nulls_by_row = selected_data.isnull().sum(axis=1)
        st.write('Total Nulls by Row:')
        st.dataframe(nulls_by_row[nulls_by_row > 0])  # Display only rows with nulls

    elif view_option == 'Each Missed Value by Row and Column':
        st.write('Missing Values by Row and Column (Red Highlight for Nulls):')
        filtered_data = selected_data.loc[selected_data.isnull().any(axis=1), selected_data.isnull().any()]
        styled_data = filtered_data.applymap(lambda x: 'Null' if pd.isnull(x) else x).style.applymap(highlight_missing)
        st.dataframe(styled_data)

# Function to check for duplicates based on selected columns
def check_duplicates(dataset, selected_columns):
    if not selected_columns:
        return pd.DataFrame()  # Return an empty DataFrame if no columns are selected

    # Combine the selected columns into a single temporary column for duplication check
    temp_dataset = dataset[selected_columns].astype(str).agg(' & '.join, axis=1)

    # Debug: Show the combined values for duplication check
    print("Combined column values for duplication check:\n", temp_dataset.head())

    # Identify duplicate entries based on the combined values
    duplicate_indices = temp_dataset[temp_dataset.duplicated(keep=False)].index

    # Debug: Print duplicate indices to verify duplication identification
    print("Duplicate indices:", duplicate_indices)

    # Get the exact duplicate rows from the original dataset based on the indices, but only the selected columns
    duplicate_rows = dataset.loc[duplicate_indices, selected_columns]

    # Debug: Print the duplicate rows for verification
    print("Duplicate rows:\n", duplicate_rows)

    return duplicate_rows



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
    if method == 'Z-score':
        # Calculate Z-scores and filter for values with abs(Z-score) > 3
        z_scores = zscore(dataset[selected_column])
        abs_z_scores = np.abs(z_scores)
        outliers = dataset[abs_z_scores > 3][selected_column]  # Return only outlier values

    elif method == 'IQR':
        # Calculate IQR and filter for values outside the 1.5 * IQR range
        Q1 = dataset[selected_column].quantile(0.25)
        Q3 = dataset[selected_column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = dataset[(dataset[selected_column] < lower_bound) | (dataset[selected_column] > upper_bound)][selected_column]

    return outliers

# Function to remove outliers using Z-score or IQR method
def remove_outliers(dataset, outliers_detected):
    return dataset[~outliers_detected]

# Load your dataset
# Assume dataset is loaded here

# Initialize outliers_detected as None at the top
outliers_detected = None

# Define mappings for consistency checks (all lowercase for case-insensitivity)
title_gender_map = {'mr': 'Male', 'Mr': 'Male','MR': 'Male','mrs': 'Female', 'Mrs': 'Female', 'MRS': 'Female', 'ms': 'Female'}
region_city_map = {
    'Addis Ababa': 'Addis Ababa', 'Tigray': 'Mekelle', 'Afar': 'Samera',
    'Amhara': 'Bahir Dar', 'Oromia': 'Adama', 'Somali': 'Somali',
    'Benishangul-Gumuz': 'Assosa', 'Southern Nations, Nationalities and Peoples Region (SNNPR)': 'Hawassa',
    'Gambella': 'Gambella', 'Harari': 'Harari', 'Dire Dawa': 'Dire Dawa'
}

# Function to check inconsistencies based on the selected type
def check_inconsistencies(dataset, selected_type):
    inconsistencies = {}

    # Perform checks specific to CBEBIRR or Digital type
    if selected_type in ['CBEBIRR', 'Digital']:
        # Convert TITLE to lowercase for case-insensitive mapping
        title_gender_inconsistencies = dataset[
            (dataset['TITLE'].str.lower().map(title_gender_map) != dataset['GENDER'])
        ]
        if not title_gender_inconsistencies.empty:
            inconsistencies['TITLE-GENDER'] = title_gender_inconsistencies[['TITLE', 'GENDER']]

    # Perform checks specific to CBEBIRR type
    if selected_type == 'CBEBIRR':
        # Check REGION and CITY consistency
        region_city_inconsistencies = dataset[
            (dataset['REGION'].map(region_city_map) != dataset['CITY'])
        ]
        if not region_city_inconsistencies.empty:
            inconsistencies['REGION-CITY'] = region_city_inconsistencies[['REGION', 'CITY']]
    
    return inconsistencies

# Function to validate data based on selected type
def validate_data(dataset, selected_type):
    if selected_type == "CBEBIRR":
        # Validate PHONE_NUMBER length (must be between 8 and 13 digits)
        invalid_phone = dataset[(dataset['PHONE_NUMBER'].astype(str).str.len() > 13) | (dataset['PHONE_NUMBER'].astype(str).str.len() < 9)]
        if not invalid_phone.empty:
            st.warning("Invalid PHONE_NUMBER entries: Phone number must be between 9 and 13 digits.")
            st.write("Entries with invalid PHONE_NUMBER length:")
            st.write(invalid_phone[['PHONE_NUMBER']])
        else:
            st.success("All PHONE_NUMBER entries are valid (between 9 and 13 digits).")

        # Calculate age and validate
        dataset['AGE'] = dataset['DATE_OF_BIRTH'].apply(lambda x: datetime.now().year - x.year if pd.notnull(x) else None)
        invalid_age = dataset[dataset['AGE'] > 120]
        if not invalid_age.empty:
            st.warning("Invalid DATE_OF_BIRTH entries: Age must not exceed 120.")
            st.write("Entries with age greater than 120:")
            st.write(invalid_age[['DATE_OF_BIRTH', 'AGE']])
        else:
            st.success("All DATE_OF_BIRTH entries are valid (age not exceeding 120).")

    elif selected_type == "Digital":
        # Calculate age and validate
        dataset['AGE'] = dataset['DATE_OF_BIRTH'].apply(lambda x: datetime.now().year - x.year if pd.notnull(x) else None)
        invalid_age = dataset[dataset['AGE'] > 120]
        if not invalid_age.empty:
            st.warning("Invalid DATE_OF_BIRTH entries: Age must not exceed 120.")
            st.write("Entries with age greater than 120:")
            st.write(invalid_age[['DATE_OF_BIRTH', 'AGE']])
        else:
            st.success("All DATE_OF_BIRTH entries are valid (age not exceeding 120).")


# Data Quality Section in Sidebar
st.sidebar.header('Data Quality Metrix/Data Profile')
# Separator for Data Overview section
st.sidebar.markdown(horizontal_line(), unsafe_allow_html=True)

# Section for Data Overview (Datatype, Mean, Median, Standard Deviation)
st.sidebar.header('0. Data Description')

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
if st.sidebar.button('Data Description'):
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
    st.write("Data Description (Datatype, Mean, Median, Std Deviation, Min, Max, Range):")
    st.dataframe(data_info)

# Separator before "1. Missing Value"
st.sidebar.markdown(horizontal_line(), unsafe_allow_html=True)

# Display all features with checkboxes
st.sidebar.header("1. View for Missing Values")
st.sidebar.write("Select Features to Include in Missing Values Analysis")
selected_features = st.sidebar.multiselect("Features", options=dataset.columns, default=[])

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
    if selected_features:  # Check if any features are selected
        display_missing_values(dataset, selected_features, view_option)
    else:
        st.warning("Please select at least one feature to analyze missing values.")

# Separator for Fill Missing Values section
st.sidebar.header("2. Fill Missing Values")

# Ensure the dataset is loaded
if 'dataset' in locals() and not dataset.empty:
    categorical_cols = dataset.select_dtypes(include=['object']).columns
    numerical_cols = dataset.select_dtypes(include=['number']).columns

    fill_options = {}

    # Categorical features fill options
    st.sidebar.subheader("Categorical Features")
    fill_all_categorical = st.sidebar.checkbox("Apply a single method to all categorical features")
    if fill_all_categorical:
        selected_method_categorical = st.sidebar.selectbox(
            "Select fill method for all categorical features",
            options=["mode", "constant", "ffill", "bfill"],
            index=0
        )
        fill_options["all_categorical"] = selected_method_categorical
        if selected_method_categorical == 'constant':
            fill_options["constant_value_categorical"] = st.sidebar.text_input(
                "Enter constant value for all categorical features", value="Unknown"
            )
    else:
        selected_categorical_features = st.sidebar.multiselect(
            "Select specific categorical features", options=categorical_cols
        )
        fill_options["selected_categorical"] = selected_categorical_features
        if selected_categorical_features:
            selected_method = st.sidebar.selectbox(
                "Select fill method for selected categorical features",
                options=["mode", "constant", "ffill", "bfill"],
                index=0
            )
            fill_options["selected_method_categorical"] = selected_method
            if selected_method == 'constant':
                fill_options["constant_value_selected_categorical"] = st.sidebar.text_input(
                    "Enter constant value for selected categorical features", value="Unknown"
                )

    # Numerical features fill options
    st.sidebar.subheader("Numerical Features")
    fill_all_numerical = st.sidebar.checkbox("Apply a single method to all numerical features")
    if fill_all_numerical:
        selected_method_numerical = st.sidebar.selectbox(
            "Select fill method for all numerical features",
            options=["mean", "median", "mode", "constant", "ffill", "bfill", "KNN"],
            index=0
        )
        fill_options["all_numerical"] = selected_method_numerical
        if selected_method_numerical == 'constant':
            fill_options["constant_value_numerical"] = st.sidebar.number_input(
                "Enter constant value for all numerical features", value=0.0
            )
    else:
        selected_numerical_features = st.sidebar.multiselect(
            "Select specific numerical features", options=numerical_cols
        )
        fill_options["selected_numerical"] = selected_numerical_features
        if selected_numerical_features:
            selected_method = st.sidebar.selectbox(
                "Select fill method for selected numerical features",
                options=["mean", "median", "mode", "constant", "ffill", "bfill", "KNN"],
                index=0
            )
            fill_options["selected_method_numerical"] = selected_method
            if selected_method == 'constant':
                fill_options["constant_value_selected_numerical"] = st.sidebar.number_input(
                    "Enter constant value for selected numerical features", value=0.0
                )

    # Apply fill method if button is clicked
    if st.sidebar.button("Fill Missing Values"):
        original_shape = dataset.shape
        dataset = fill_missing_values(dataset, fill_options)
        st.write(f"Missing values filled. Original shape: {original_shape}, New shape: {dataset.shape}")

        # Allow the user to download the processed data
        csv_data = convert_df_to_csv(dataset)
        st.download_button(
            label="Download Processed Data as CSV",
            data=csv_data,
            file_name='processed_data.csv',
            mime='text/csv'
        )
else:
    st.write("No data available. Please upload a valid dataset.")
# Separator for Duplicates section
st.sidebar.markdown(horizontal_line(), unsafe_allow_html=True)

# Sidebar Section to Check for Duplicates
st.sidebar.header('3. Duplicate Value')

# Dropdown to select the type of check
selected_type = st.sidebar.selectbox('Select Type to Check for Duplicates', ['Credit', 'Digital', 'CBEBIRR', 'FCY'])

# Allow user to multi-select columns for duplication check
if dataset is not None and not dataset.empty:
    selected_columns = st.sidebar.multiselect(
        f'Select columns for {selected_type} duplicate check',
        options=dataset.columns,  # Allow selection from all columns in the dataset
    )
else:
    selected_columns = []

# Button to check for duplicate values
if st.sidebar.button('Check for Duplicates'):
    if dataset is not None and not dataset.empty:
        # Debug: Print the dataset to verify the structure
        print("Dataset columns:", dataset.columns)
        
        duplicates_df = check_duplicates(dataset, selected_columns)
        duplicate_count = duplicates_df.shape[0]  # Count the number of duplicated rows

        if duplicate_count > 0:
            st.write(f"Total Duplicated Rows for {selected_type}: {duplicate_count}")
            st.write("Here are the exact duplicate rows (only selected columns) for cross-checking:")
            st.dataframe(duplicates_df)  # Display only the selected columns for duplicated rows
        else:
            st.write("No duplicated rows found.")
    else:
        st.write("No data available. Please upload a valid dataset.")


        
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

# Dropdown to select the method for outlier detection
outlier_method = st.sidebar.radio('Select Outlier Detection Method', ['Z-score', 'IQR'])

# Button to detect outliers
if st.sidebar.button('Detect Outliers'):
    if outlier_column:
        outliers = detect_outliers(dataset, outlier_method, outlier_column)
        
        st.write(f"Outliers detected in '{outlier_column}' using {outlier_method}:")

        # Display exact outlier values if any found
        if not outliers.empty:
            st.write("Exact outlier values from the dataset:")
            st.dataframe(outliers.reset_index().rename(columns={outlier_column: 'Outlier Value'}))
        else:
            st.write("No outliers detected.")

        # Display boxplot for visualizing outliers
        st.write(f"Box Plot for '{outlier_column}':")
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
#  separator
st.sidebar.markdown(horizontal_line(), unsafe_allow_html=True)

# Section for Data Inconsistency 
st.sidebar.header('7. Data Inconsistency')
# Dropdown for selecting the type
selected_type = st.sidebar.selectbox('Select Type', options=['CBEBIRR', 'Credit', 'Digital', 'FCY'])
# Button to check inconsistencies based on selected type
if st.sidebar.button('Check Inconsistencies'):
    # Display specific inconsistencies based on type
    specific_inconsistencies = check_inconsistencies(dataset, selected_type)
    if specific_inconsistencies:
        st.write(f'{selected_type}-Specific Inconsistencies:')
        for key, df in specific_inconsistencies.items():
            st.write(f"{key} Inconsistencies:")
            st.write(df)
    else:
        st.write(f"No {selected_type}-specific inconsistencies found.")
# Final separator
st.sidebar.markdown(horizontal_line(), unsafe_allow_html=True)

# Button to check validation
# Sidebar for dataset type selection
# Sidebar for dataset type selection
st.sidebar.header('8. Data Validity')
dataset_type = st.sidebar.selectbox("Select Dataset Type", options=["CBEBIRR", "Credit", "Digital", "FCY"])

# Validate button in the sidebar
if st.sidebar.button("Check Validity"):
    # Ensure the dataset is already loaded in the environment
    try:
        validate_data(dataset, dataset_type)
    except NameError:
        st.error("Dataset not found. Please load your dataset before validation.")
# Final separator
st.sidebar.markdown(horizontal_line(), unsafe_allow_html=True)
####################### End of Code ############################



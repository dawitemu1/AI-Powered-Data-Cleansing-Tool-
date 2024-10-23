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
st.markdown('<h1 style="text-align: center;">Explore Loan EDA Dynamics CBE</h1>', unsafe_allow_html=True)

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


@st.cache_data
def plot_categorical_feature(data, categorical_feature, figsize1=(8, 5), figsize2=(6, 6), figsize3=(12, 8)):
    plt.style.use('fivethirtyeight')
    sns.set(style="whitegrid")
    # Convert the 'edhs_year' column to string
    data['year'] = data['year'].astype(str)

    try:
        # Create a figure with a grid layout
        fig = plt.figure(figsize=(figsize1[0] + figsize2[0], max([figsize1[1], figsize2[1]])))
        gs = gridspec.GridSpec(1, 2, width_ratios=[figsize1[0], figsize2[0]])

        # First Subplot
        ax0 = plt.subplot(gs[0])
        ax0.set_title(f'Distribution by {categorical_feature}')
        sns.countplot(data=data, x=categorical_feature, order=data[categorical_feature].value_counts().index, ax=ax0)
        ax0.set_xlabel(categorical_feature)
        ax0.set_ylabel("Count")
        ax0.tick_params(axis='x', rotation=80)

        # Add labels on top of the bars
        for p in ax0.patches:
            ax0.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom', fontsize=8, color='black')

        # Second(pie chart) Subplot
        ax2 = plt.subplot(gs[1])
        ax2.set_title(f'Distribution by {categorical_feature}')
        data[categorical_feature].value_counts().plot.pie(autopct='%1.1f%%',  shadow=True, ax=ax2)

        # Another Bar chart
        fig2, ax1 = plt.subplots(figsize=figsize3)
        ax1.set_title(f'Distribution by {categorical_feature} and Loan year')
        sns.countplot(data=data, x=categorical_feature, hue='year', order=data[categorical_feature].value_counts().index, ax=ax1)
        ax1.set_xlabel(categorical_feature)
        ax1.set_ylabel("Count")
        ax1.tick_params(axis='x', rotation=80)

        # Add labels on top of the bars
        for p in ax1.patches:
            ax1.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom', fontsize=8, color='black')

        # Show the figures
        st.write(fig)
        st.pyplot(fig2)

    except Exception as e:
        st.error(f"An error occurred while plotting: {e}")


# Function to display frequency distribution table
@st.cache_data
def display_frequency_distribution(dataset, categorical_feature):
    st.write('===================================================================')
    st.write(f'     Percentage distribution of {categorical_feature} feature category based on different CBE_Loan year')
    st.write('===================================================================')

    result = pd.crosstab(
        index=dataset[categorical_feature],
        columns=dataset['year'],
        values=dataset['LOAN_STATUS'],
        aggfunc='count',
        margins=True,  # Added 'margins' for total row/column
        margins_name='Total'  # Custom name for the 'margins' column/row
    )
    result['|'] = '|'

    for year in result.columns[:6]:  # Exclude the last column 'Total'
        result[f'{year}(%)'] = (result[year] / result[year]['Total']) * 100

    # Round the percentage values to 1 decimal place
    result = result.round(1).fillna(0)

    # Display the table using st.dataframe
    st.dataframe(result)
    st.write('====================================================================')

######################## Correlation Analysis ##############################################
# Function to generate the correlation analysis
@st.cache_data
def correlation_analysis(dataset, selected_features, selected_years):
    # Filter dataset based on selected loan years
    selected_data = dataset[dataset['year'].isin(selected_years)]

    # Filter dataset based on selected numerical features
    selected_data = selected_data[selected_features]

    # Calculate the correlation matrix
    corr = selected_data.corr()

    # Calculate p-values
    p_values = pd.DataFrame(index=corr.index, columns=corr.columns)

    for i in range(len(corr)):
        for j in range(len(corr.columns)):
            coef, p_value = pearsonr(selected_data.iloc[:, i], selected_data.iloc[:, j])
            p_values.iloc[i, j] = p_value

    # Round correlation matrix to three decimal places
    corr = corr.round(3)

    # Increase the figure size
    plt.figure(figsize=(16, 12))

    # Plot the correlation heatmap
    sns.heatmap(corr, fmt=".3f", cmap='Blues', cbar_kws={'shrink': 0.8})

    # Manually add text annotations for both correlation and p-values
    for i in range(len(corr)):
        for j in range(len(corr.columns)):
            text = plt.text(j + 0.5, i + 0.5, f"{corr.iloc[i, j]:.3f}\n(p={p_values.iloc[i, j]:.3f})",
                            ha='center', va='center', color='black', fontsize=10)

    plt.title(f"Correlation Plot of selected Features for Loan Years {', '.join(map(str, selected_years))}")
    st.pyplot(plt)
#########################################################################

# # Additional styling for the overview section
# st.markdown(
#     """
#     ## Welcome to Loan Status EDA Tool!

#     Explore the dynamics of Loan Status in CBE across different Loan years. This interactive tool, powered by exploratory data analysis (EDA), offers insights into key trends.

#     ### What You Can Do:
#     1. Feature Distribution: Examine feature distributions on both aggregated and by specific loan year.
#     2. Bivariate Analysis: Examine feature Selected feature with Loan_status aggregated  year
#     3. Correlation Analysis: Understand feature correlations for any CBE year.
#     4. Generete report: generate report yearly, monthly, weekly, daily for seelected features with loan_status 
#     5. Generate Quarterly Report: Generate reprot Quarterly with specfic year and feature.
#     6. In single year report and quartly report model show total loan_status for selected granurilty and quarter

#     Dive into the rich data of CBE from 2014 to 2024, interact, and uncover valuable insights!
#     """
# )

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

# Sidebar for selecting parametersNavigation_Menu',
st.sidebar.header('Parameters')

####################### 1. Feature Distribution of UFM ##################################
st.sidebar.markdown('### Feature Distribution')
# Allow the user to select a feature
selected_feature = st.sidebar.selectbox('Select Feature', allowed_features)

# Display distribution plots and tables based on the selected feature
if st.sidebar.button('Show Distribution'):
    st.subheader(f'Frequency Distribution of {selected_feature}')
    display_frequency_distribution(dataset, selected_feature)
    st.subheader(f'Distribution of {selected_feature}')
    plot_categorical_feature(dataset, selected_feature)

# Separator
st.sidebar.markdown(horizontal_line(), unsafe_allow_html=True)



####################### 2. Correlation Analysis ###################################################
st.sidebar.header('Correlation Analysis')
# Filter numerical features only
numerical_features = dataset.select_dtypes(include=['number']).columns

# Allow the user to select the features
selected_features = st.sidebar.multiselect('Select Features for Correlation', numerical_features, key='features')

# Allow the user to select multiple EDHS years using a multiselect
selected_years = st.sidebar.multiselect('Select Loan Years', dataset['year'].unique(), key='years')

# Button to generate correlation matrix
if st.sidebar.button('Generate Correlation Matrix'):
    correlation_analysis(dataset, selected_features, selected_years)

# Separator
st.sidebar.markdown(horizontal_line(), unsafe_allow_html=True)


####################### 3. Data quality ###################################################
import pandas as pd
import streamlit as st
import missingno as msno
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
import io

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

# Function to display missing values
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
        st.pyplot(fig)  # Render the plot in Streamlit

# Load your dataset
# Assume dataset is loaded here

# Separator
# st.sidebar.markdown(horizontal_line(), unsafe_allow_html=True)

# Data Quality Section in Sidebar
st.sidebar.header('Data Quality')
st.sidebar.header('1. Missing Value')
# Allow the user to choose the method to view missing values
view_option = st.sidebar.radio('Select View for Missing Values', 
                                ['Total Null Values', 
                                 'Total Nulls by Table', 
                                 'Total Nulls by Percentage', 
                                 'Bar Graph of Missing Values'])

# Button to display missing values based on selected option
if st.sidebar.button('Show Missing Values'):
    display_missing_values(dataset, view_option)

# Separator for Fill Missing Values section
st.sidebar.markdown(horizontal_line(), unsafe_allow_html=True)

# New Section for Filling Missing Values
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

# Final separator
st.sidebar.markdown(horizontal_line(), unsafe_allow_html=True)


####################### End of Code ############################
    
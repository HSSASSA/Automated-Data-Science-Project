import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from gemini_eda import perform_eda
import os
# Set page configuration
st.set_page_config(page_title="Gemini EDA", page_icon="📊", layout="wide")

# Custom CSS to improve the appearance
st.markdown("""
    <style>
    .main > div {
        padding: 2rem;
    }
    .stPlotly {
        margin: 1rem 0;
    }
    h3 {
        margin-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)


path = os.path.join(os.getcwd(),"Data")
if os.path.isdir(path):
    st.title("📊 :blue[Gemini EDA - Exploratory Data Analysis with AI]")
    file = os.listdir(path)
    if file:
        df = pd.read_csv(os.path.join("Data", file[0])) #,index_col=0       
        
        # Data Preview Section
        st.header("📋 Data Preview")
        st.write("Here are the first few rows of your datase    t:")
        st.dataframe(df.head(), use_container_width=True)
        
        if st.button("🔍 Perform EDA"):
            with st.spinner("🤖 AI is analyzing your data..."):
                eda_results = perform_eda(df)
            
            # Overall Insights Section
            st.header("🔎 Overall Insights")
            st.markdown(eda_results.get("overall_insights", "No insights available."))
            
            # Basic Statistics Section
            st.header("📈 Basic Statistics")
            st.write("Summary statistics for numerical columns in your dataset:")
            st.dataframe(df.describe(), use_container_width=True)
            
            # Data Types Section
            st.header("🏷️ Data Types")
            st.write("Overview of the data types in your dataset:")
            col1, col2 = st.columns([2, 1])
            with col1:
                st.dataframe(pd.DataFrame(df.dtypes, columns=['Data Type']), use_container_width=True)
            with col2:
                st.markdown(eda_results.get("data_types", "No data type information available."))
            
            # Missing Values Section
            st.header("❓ Missing Values Analysis")
            missing_data = pd.DataFrame({
                'Missing Values': df.isnull().sum(),
                'Percentage': (df.isnull().sum() / len(df)) * 100
            })
            col1, col2 = st.columns([2, 1])
            with col1:
                st.dataframe(missing_data, use_container_width=True)
            with col2:
                st.markdown(eda_results.get("missing_values", "No missing value analysis available."))
            
            # Distribution Section
            st.header("📊 Data Distribution")
            st.markdown(eda_results.get("distribution", "No distribution analysis available."))
            
            # Numerical Distributions
            numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
            if len(numerical_cols) > 0:
                st.subheader("Numerical Variables Distribution")
                for col in numerical_cols:
                    col1, col2 = st.columns([3, 2])
                    with col1:
                        fig, ax = plt.subplots(figsize=(8, 5))
                        sns.histplot(data=df, x=col, kde=True)
                        plt.title(f'Distribution of {col}')
                        st.pyplot(fig)
                        plt.close()
                    with col2:
                        st.write(f"**Statistics for {col}:**")
                        st.write(df[col].describe())
            
            # Categorical Distributions
            categorical_cols = df.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                st.subheader("Categorical Variables Distribution")
                for col in categorical_cols:
                    col1, col2 = st.columns([3, 2])
                    with col1:
                        fig, ax = plt.subplots(figsize=(8, 5))
                        df[col].value_counts().plot(kind='bar')
                        plt.title(f'Distribution of {col}')
                        plt.xticks(rotation=45)
                        st.pyplot(fig)
                        plt.close()
                    with col2:
                        st.write(f"**Value counts for {col}:**")
                        st.write(df[col].value_counts())
            
            # Correlation Analysis
            if len(numerical_cols) > 1:
                st.header("🔗 Correlation Analysis")
                st.markdown(eda_results.get("correlations", "No correlation analysis available."))
                
                col1, col2 = st.columns([3, 2])
                with col1:
                    # Correlation Matrix
                    fig, ax = plt.subplots(figsize=(8, 6))
                    correlation_matrix = df[numerical_cols].corr()
                    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
                    plt.title('Correlation Matrix')
                    st.pyplot(fig)
                    plt.close()
                with col2:
                    st.write("**Key Correlations:**")
                    st.write(correlation_matrix.unstack().sort_values(ascending=False).drop_duplicates().head(5))
                
                # Pairplot with smaller size
                st.subheader("Pairwise Relationships")
                fig = sns.pairplot(df[numerical_cols], height=2)
                st.pyplot(fig)
                plt.close()
            
            # AI Insights Section
            st.header("🤖 AI Insights")
            st.subheader("Key Patterns and Anomalies")
            st.markdown(eda_results.get("patterns_anomalies", "No patterns or anomalies identified."))
            
            st.subheader("Business Recommendations")
            st.markdown(eda_results.get("business_recommendations", "No business recommendations available."))
            
            # Download Section
            st.header("📥 Download Analysis")
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Data as CSV",
                data=csv,
                file_name="analyzed_data.csv",
                mime="text/csv"
            )
else :
    st.warning("No data file found in the Data directory. Please upload a CSV file.")    
    st.page_link("pages/main.py",icon='🏠')
# Sidebar
st.sidebar.header("About")
st.sidebar.info(
    """
    You are using Gemini's AI agent to perform 
    Exploratory Data Analysis on your uploaded CSV files.
    
    Features:
    - Data Preview
    - Basic Statistics
    - Distribution Analysis
    - Correlation Analysis
    - AI-powered Insights
    """
)
st.sidebar.image("https://360digitmg.com/uploads/blog/7cb311c355ff80bb4a34e1faba04fb2e.png")


## Use the below code for the EDA without the gemini agent
# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# # from ydata_profiling import ProfileReport
# # from streamlit_pandas_profiling import st_profile_report
# import os

# st.title("Automated Exploratory Data Analysis")
# st.sidebar.image("https://360digitmg.com/uploads/blog/7cb311c355ff80bb4a34e1faba04fb2e.png")

# path = os.path.join(os.getcwd(),"Data")
# #Check if there is a file in the directory "Sweetviz" = if it exists then it contains a file
# if os.path.isdir(path):
#     file = os.listdir(path)
#     if file:
#         df = pd.read_csv(os.path.join("Data", file[1]),index_col=0)
#         st.write("Data Preview", df.head())
#         # Display basic statistics
#         st.write("Basic Statistics", df.describe())

#         # Correlation heatmap
#         st.write("Correlation Heatmap")
#         fig, ax = plt.subplots()
#         sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
#         st.pyplot(fig)

#         # Pairplot
#         st.write("Pairplot")
#         sns.pairplot(df)
#         st.pyplot()

#         # # Generate and display profile report
#         # profile = ProfileReport(df, title="Pandas Profiling Report", explorative=True)
#         # st_profile_report(profile)
#     else:
#         st.warning("No data file found in the 'Data' directory.")
# else :
#     st.warning("No data file found in the Data directory. Please upload a CSV file.")    
#     st.page_link("pages/main.py",icon='🏠')
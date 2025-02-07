import google.generativeai as genai
import pandas as pd
import os
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# Configure the Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def perform_eda(df: pd.DataFrame) -> dict:
    # Get basic statistics
    stats = df.describe().to_string()
    
    # Get data info
    # buffer = pd.StringIO()
    # df.info(buf=buffer)
    # data_info = buffer.getvalue()
    data_info = df.info()
    print (data_info)
    # Prepare the prompt for Gemini
    prompt = f"""
    Analyze this dataset and provide insights. Here's the information:
    
    Basic Statistics:
    {stats}
    
    Data Info:
    {data_info}
    
    Please provide a detailed analysis with the following sections:
    1. Overall Insights: Summarize the key findings from the dataset
    2. Data Types: Explain the data types and their implications
    3. Missing Values: Analyze any patterns in missing data
    4. Distribution: Describe the distribution of numerical and categorical variables
    5. Correlations: Explain relationships between variables
    6. Patterns and Anomalies: Highlight important patterns and potential outliers
    7. Business Recommendations: Provide actionable insights based on the analysis
    
    Format each section separately and provide clear, concise insights.
    """

    # Set up the model
    model = genai.GenerativeModel('gemini-pro')
    
    # Generate the EDA
    response = model.generate_content(prompt)
    
    # Split the response into sections
    sections = response.text.split('\n\n')
    
    # Create a dictionary with default values
    eda_results = {
        "overall_insights": "No overall insights available.",
        "data_types": "",
        "missing_values": "No missing value analysis available.",
        "distribution": "No distribution analysis available.",
        "correlations": "No correlation analysis available.",
        "patterns_anomalies": "No patterns or anomalies identified.",
        "business_recommendations": "No business recommendations available."
    }
    
    # Update the dictionary with actual results if available
    section_keys = list(eda_results.keys())
    for i, section in enumerate(sections):
        if i < len(section_keys):
            eda_results[section_keys[i]] = section.strip()
    
    return eda_results

# Streamlit Data Transformation and Model Training App

This project is a Streamlit application that allows users to upload a CSV file, preprocess the data, visualize it, make EDA with Gemini's agent on it, train machine learning models and make predictions using the best model.

## Project Structure

```
streamlit-app
├── src
│   ├── app.py               # Main entry point for the Streamlit application
│   ├── data_transformation.py # Contains DataTransformation class for data preprocessing
│   ├── model_training.py     # Functions for training and evaluating machine learning models
│   ├── utils.py              # Utility functions for data visualization and logging
│   └── pages
│       └── main.py         # The main page where you upload the csv file
│       └── transform.py         # Page for transforming data
│       └── EDA.py         # Exploratory Data Analysis with gemini's LLM
│       └── visualize.py         # Page data visualisation 
│       └── ML_Pipeline.py         # Where you can train machine learning models 
│       └── predict.py         # Where you make predictions
│   └── types
│       └── index.py         # Custom types and interfaces for type safety
├── requirements.txt          # Python dependencies for the project
└── README.md                 # Documentation for the project
```

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd streamlit-app
   ```

2. **Create a virtual environment (optional but recommended):**
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required packages:**
   ```
   pip install -r requirements.txt
   ```

## Usage Guidelines

1. **Run the Streamlit application:**
   ```
   streamlit run src/app.py
   ```

2. **Upload a CSV file:**
   - Use the file uploader in the application to upload your dataset.

3. **Data Transformation:**
   - The application will handle missing values, encode categorical columns, and scale numerical columns.

4. **Model Training:**
   - The application will train various machine learning models and display the best model for making predictions.

## Overview of Functionality

- **Data Upload:** Users can upload CSV files containing their datasets.
- **Data Preprocessing:** The application preprocesses the data by handling missing values, encoding categorical features, and scaling numerical features.
- **Model Training:** Various machine learning models are trained on the preprocessed data, and the best-performing model is identified.
- **Results Display:** The application displays the transformed data and the results of the model training process.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
# Adjust the project root path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from data_transformation import DataTransformation
from model_training import ModelTrainer
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="ML_Pipeline", page_icon="👨🏻‍🌾", layout="wide")
st.sidebar.image("https://blog.talent500.co/wp-content/uploads/2025/01/Machine_Learning-1024x819-1.jpg")
path = os.path.join(os.getcwd(),"Data")

if os.path.isdir(path):
    task_type = st.selectbox("Select the task type", ["👇🏼", "Classification", "Regression"])
    file = os.listdir(path)
    if file:
        # df = pd.read_csv(os.path.join("Data", file[0]))#,index_col=0
        # Select task type
            if task_type == "Classification":
                task = "Classification"
                df = pd.read_csv(os.path.join("Data", file[0])) # take the first file (original data)
                # Select target column
                target_column = st.selectbox("Select the target column (y)", df.columns)
                # Select feature columns
                feature_columns = st.multiselect("Select the feature columns (X)", df.columns, default=[col for col in df.columns if col != target_column])
            elif task_type == "Regression":
                task = "Regression"
                try:
                    df = pd.read_csv(os.path.join("Data", file[1]),index_col=0) #take the transformed data (the second file)
                except Exception as e:
                    st.warning("Please transform the data first.")    
                    st.page_link("pages/transform.py",icon='🏠')
                    # st.error(f"Error reading the transformed data: {e}")
                # Select target column
                target_column = st.selectbox("Select the target column (y)", df.columns)
                # Select feature columns
                feature_columns = st.multiselect("Select the feature columns (X)", df.columns, default=[col for col in df.columns if col != target_column])
            else:
                task = None
                st.warning("Please select a task type.")

            go = False
            if 'target_column' in locals() and 'feature_columns' in locals():
                if target_column and feature_columns:
                    # Spliting data into X and y
                    X = df[feature_columns]
                    y = df[target_column]
                    # Ensuring X and y have consistent lengths
                    if len(X) != len(y):
                        st.error("Error: Inconsistent number of samples between X and y.")
                    # Train-test spliting
                    try:
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    except Exception as e:
                        st.error(f"Error during train-test split: {e}")

                    if task_type == "Classification":
                        # Ensuring y is categorical
                        if y.dtype == 'float64' or y.dtype == 'int64':
                            st.error("Error: Target variable for classification should be categorical.")
                        else:
                            st.success("Classification task selected.")
                            go = True
                    elif task_type == "Regression":
                        # Ensuring y is continuous
                        if y.dtype == 'object':
                            st.error("Error: Target variable for regression should be continuous.")
                        else:
                            st.success("Regression task selected.")
                            go = True
                        for feature in X.columns:
                            if X[feature].dtype == 'object':
                                st.error(f"Error: Feature {feature} is categorical. Please encode it.")

                    if go:
                        path_to_features = os.path.join(os.getcwd(),"feat")
                        os.makedirs(path_to_features,exist_ok=True)
                        X.to_csv("feat/features_df.csv",index=False)

                        st.subheader(":blue[Let's train some models!] ")
                        if st.button("Run 🚀"):
                            with st.spinner("Training models..."):

                                try:
                                    # Initializing ModelTrainer object with task_type
                                    trainer = ModelTrainer(task_type)
                                    train_array = np.column_stack((X_train, y_train))
                                    test_array = np.column_stack((X_test, y_test))
                                    model_performances = trainer.initiate_model_training(train_array=train_array,
                                                                                    test_array=test_array)

                                    # st.write("Best model trained : ", best_model_name)
                                    st.subheader(":blue[Model Performances:]")
                                    st.write(model_performances)
                                    st.page_link("pages/predict.py",icon='✨')

                                except Exception as e:
                                    st.error(f"Error in model training: {str(e)}")

else :
    st.warning("No data file found in the Data directory. Please upload a CSV file.")    
    st.page_link("pages/main.py",icon='🏠')
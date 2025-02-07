import streamlit as st
import os
import sys
import pandas as pd
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from data_transformation import DataTransformation

st.set_page_config(page_title="transform", page_icon="👷🏻‍♂️", layout="wide")
st.sidebar.image("https://rivery.io/wp-content/uploads/2023/05/Data-transformation@2x.png")

path = os.path.join(os.getcwd(),"Data")
if os.path.isdir(path):
    st.subheader("Transform, visualize and train machine learning models on your own data!")
    m = ["👇🏼","Transform the uploaded data", "Upload new data"]
    ch = st.selectbox("Choose",m)
    #Check if there is a file in the directory "Sweetviz" = if it exists then it contains a file
    if ch == "Transform the uploaded data":
        if os.path.isdir(path):
            file = os.listdir(path)
            if file:
                df = pd.read_csv(os.path.join("Data", file[0]))
                st.write("Original Data: ", df)
                try:
                    transformer = DataTransformation()
                    df_transformed = transformer.transform_data(df)
                    df_transformed.to_csv("Data/transformed_df.csv",index=True)
                    st.success("Transformed data saved successfully in the path: " + path)
                    st.write("Transformed Data", df_transformed)
                    st.subheader("👇🏼:blue[Visualize the data]")
                    st.page_link("pages/visualize.py",icon='👨🏻‍💻')
                    st.subheader("👇🏼:blue[Explore and Analyse the data!]")
                    st.page_link("pages/EDA.py",icon='🕵🏻‍♂️')
                    st.subheader("👇🏼:blue[Train machine learning models on it!] ")
                    st.page_link("pages/ML_Pipeline.py",icon='👨🏻‍🌾')
                except Exception as e:
                    st.write(f"Error transforming data: {e}")

    if ch == "Upload new data":
        st.subheader("Please upload your CSV file")
        uploaded_file = st.file_uploader("Upload CSV",type=['csv'])
        if uploaded_file is not None:
            file_info = {"file name" : uploaded_file.name,
                        "file size" : uploaded_file.size}
            st.write("File info : " , file_info)

            # path2 = os.path.join(os.getcwd(),"Data","New data")
            # os.makedirs(path2,exist_ok=True)
            # path_to_data2 = os.path.join(path2, uploaded_file.name)
            # with open(path_to_data2 , 'wb') as f:
            #     f.write(uploaded_file.getbuffer())
            #     st.success("file saved successfully in the path : " + path2)
            st.write("Original data")
            df = pd.read_csv(uploaded_file,index_col=0)
            st.dataframe(df)
            # st.dataframe(df.head())
            if st.button("Transform this data"):
                try:
                    transformer = DataTransformation()
                    df_transformed = transformer.transform_data(df)
                    st.write("Transformed data")
                    st.dataframe(df_transformed)
                    # df_transformed.to_csv(f"{path2}/transformed_df.csv",index=True)
                
                except Exception as e:
                        st.write(f"Error transforming data: {e}")
        
            st.subheader("👇🏼:blue[Visualize the data]")
            st.page_link("pages/visualize.py",icon='👨🏻‍💻')
            st.subheader("👇🏼:blue[Explore and Analyse the data!]")
            st.page_link("pages/EDA.py",icon='🕵🏻‍♂️')
            st.subheader("👇🏼:blue[Train machine learning models on it!] ")
            st.page_link("pages/ML_Pipeline.py",icon='👨🏻‍🌾')
else :
    st.warning("No data file found in the Data directory. Please upload a CSV file.")    
    st.page_link("pages/main.py",icon='🏠')
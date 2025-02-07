import streamlit as st
import sys
import os
# Adjust the project root path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
import streamlit.components.v1 as components
import pandas as pd

st.set_page_config(page_title="main", page_icon="🏠")
st.sidebar.image("https://cdn.prod.website-files.com/65e7297194523c404b923b44/662e717bbeab55128dd63396_how-to-create-welcome-page-1.jpeg", caption="Home Page")
# Function to load HTML content from a file
def load_html(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# load Style css
with open('templates/style.css')as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html = True)

html_temp = """
<div style="background-color:royalblue;padding:30px;border-radius:20px;border:3px outsed red">
<h1 style="color:white;text-align:center;">Transform, Visualize, and train robust machine learning models on your own Data!</h1>
</div>
"""
components.html(html_temp)
# st.subheader("Upload and work with your own data!")
uploaded_file = st.file_uploader(":blue[Upload a CSV file to get started]", type=['csv'])
if uploaded_file is not None:
    path = os.path.join(os.getcwd(),"Data")
    os.makedirs(path,exist_ok=True)
    path_to_data = os.path.join(path, uploaded_file.name)
    with open(path_to_data, 'wb') as f:
        f.write(uploaded_file.getbuffer())
        st.success("File saved successfully in the path: " + path)
    df = pd.read_csv(uploaded_file)
    file_info = {"file name" : uploaded_file.name,
                        "file size" : uploaded_file.size}
    st.subheader("File info : ")
    st.write(file_info)
    st.subheader("Content:")
    st.dataframe(df)

    components.html("<h1 style='color:blue;'>👇🏼 See what you can do with this data!</h1>")
    st.write("1. Transform the data ")
    st.page_link("pages/transform.py",icon='👷🏻‍♂️')
    st.write("2. Visualize the data ")
    st.page_link("pages/visualize.py",icon='👨🏻‍💻')
    st.write("3. Explore and Analyse the data! ")
    st.page_link("pages/EDA.py",icon='🕵🏻‍♂️')
    st.write("4. Train ML models on it! ")
    st.page_link("pages/ML_Pipeline.py",icon='👨🏻‍🌾')
    st.write("5. Make predictions! ")
    st.page_link("pages/predict.py",icon='✨')
import streamlit as st
import pandas as pd
import sys
import os
import numpy as np
# Adjust the project root path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
import streamlit.components.v1 as components


# Function to load HTML content from a file
def load_html(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def main():
    # with open('templates/style.css')as f:
    #     st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html = True)
    # html_temp = """
    # <div style="background-color:royalblue;padding:10px;border-radius:10px">
    # <h1 style="color:white;text-align:center;">Automated Machine Learning Project App</h1>
    # </div>
    # """
    # # components.html("<p style='color:red;'> Streamlit application</p>")
    # components.html(html_temp)
    components.html(load_html("templates/home.html"),height=600)
    # components.html("<p style='color:Cyan;'> Upload and work with your own data!</p>")
    st.subheader(":blue[Let's go!🏃‍♂️‍➡️]")
    st.page_link("pages/main.py",icon='🥷🏻')
    
if __name__ == "__main__":
    main()
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(page_title="visualize", page_icon="👨🏻‍💻", layout="wide")
st.sidebar.image("https://images.ctfassets.net/te2janzw7nut/6VoWjuosDdasNydaWot2ZM/34a7893a8c72203dcbee5d452c70e36b/data-visualization.jpg?w=500&h=332&fl=progressive&q=100&fm=jpg")

# Load data
if os.path.isdir("Data"):
    file = os.listdir("Data")
    if file:
        df = pd.read_csv(os.path.join("Data", file[0]))
        st.subheader(":blue[Your data ⬇️]")
        st.write(df)
        # Select columns for x and y axes
        x_axis = st.selectbox("Choose a column for the X axis",df.columns)
        y_axis = st.selectbox("Choose a column for the Y axis", df.columns)

        # Select type of visualization
        chart_type = st.selectbox("Choose the type of chart", ["Select the type of visual you want 👇🏼", "Line Chart", "Bar Chart", "Scatter Plot"])

        # Generate chart based on user input
        if chart_type == "Line Chart":
            st.line_chart(df[[x_axis, y_axis]].set_index(x_axis))
        elif chart_type == "Bar Chart":
            st.bar_chart(df[[x_axis, y_axis]].set_index(x_axis))
        elif chart_type == "Scatter Plot":
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x=x_axis, y=y_axis, ax=ax)
            st.pyplot(fig)
else :
    st.warning("No data file found in the Data directory. Please upload a CSV file.")    
    st.page_link("pages/main.py",icon='🏠')
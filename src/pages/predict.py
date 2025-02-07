import streamlit as st
import pandas as pd
import os
import sys
import pickle
import numpy as np

# Adjust the project root path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

st.set_page_config(page_title="predict", page_icon="🤔", layout="wide")
st.sidebar.image("https://builtin.com/sites/www.builtin.com/files/2023-01/machine-learning-stock-prediction.jpg")

# Load the model
if os.path.isdir("artifacts"):
    data = pd.read_csv(os.path.join("Data", os.listdir(os.path.join(os.getcwd(),"Data"))[0]))
    st.dataframe(data)
    file = os.listdir("artifacts")
    if file:
        model_path = os.path.join("artifacts", "model.pkl")
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        st.title(":blue[Make Predictions]")

        # Upload data
        if os.path.isdir("feat"):
            file = os.listdir("feat")
            if file:
                features = pd.read_csv(os.path.join("feat", file[0]))

                    # User input for features
                user_input = {}
                for feature in features.columns:
                    user_input[feature] = st.number_input(f"Enter value for {feature}", value=0.0)

                # Convert user input to DataFrame
                input_df = pd.DataFrame([user_input])

                # Validate user input
                def validate_input(input_df, df):
                    numeric_df = df.select_dtypes(include=[np.number])
                    z_scores = np.abs((input_df - numeric_df.mean()) / numeric_df.std())
                    outliers = z_scores > 3  # Z-score threshold for outliers
                    if outliers.any().any():
                        return False, "Input contains outliers."
                    return True, ""

                is_valid, message = validate_input(input_df, features)
                if not is_valid:
                    st.warning(message)
                else:
                    # Make prediction
                    if st.button("Predict 🤔"):
                        prediction = model.predict(input_df)
                        st.subheader("⬇️")
                        st.subheader(f":blue[{prediction[0]}]")

            else:
                st.warning("No Data found, please upload your data!")
        else:
            st.warning("No Data directory found. Please ensure the 'Data' directory exists.")
else:
    st.warning("Model file not found. Please ensure 'model.pkl' is in the 'artifacts' folder.")
    st.page_link("pages/main.py", icon='🏠')


# import streamlit as st
# import pandas as pd
# import os
# import sys
# import pickle
# import numpy as np
# # Adjust the project root path
# project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(project_root)
# from pages.ML_Pipeline import task, features

# st.sidebar.image("https://builtin.com/sites/www.builtin.com/files/2023-01/machine-learning-stock-prediction.jpg")
# # Load the model
# if os.path.isdir("artifacts"):
#     file = os.listdir("artifacts")
#     if file:
#         model_path = os.path.join("artifacts", "model.pkl")
#         with open(model_path, 'rb') as file:
#             model = pickle.load(file)
#         st.title(":blue[Make Predictions]")

#         # upload data 
#         if os.path.isdir("Data"):
#             file = os.listdir("Data")
#             if file:
#                 df = pd.read_csv(os.path.join("Data", file[0]))
#                 st.subheader("You  are using "+ task + " models")
#                 st.write("Data Preview", df)


#                 # Get feature names
#                 # feature_names = df.columns.tolist()
#                 feature_names = features.columns.tolist()
#                 # User input for features
#                 user_input = {}
#                 # for feature in feature_names[:-1]:
#                 #     user_input[feature] = st.number_input(f"Enter value for {feature}", value=0.0)
#                 for feature in features:
#                     user_input[feature] = st.number_input(f"Enter value for {feature}", value=0.0)

#                 # Convert user input to DataFrame
#                 input_df = pd.DataFrame([user_input])
#                 # Validate user input
#                 def validate_input(input_df, df):
#                     numeric_df = df.select_dtypes(include=[np.number])
#                     z_scores = np.abs((input_df - numeric_df.mean()) / numeric_df.std())
#                     # z_scores = np.abs((input_df - df.mean()) / df.std())
#                     outliers = z_scores > 3  # Z-score threshold for outliers
#                     if outliers.any().any():
#                         return False, "This is an outlier."
#                     return True, ""

#                 is_valid, message = validate_input(input_df, df)
#                 if is_valid:
#                     # Make prediction
#                     if st.button("Predict 🤔"):
#                         prediction = model.predict(input_df)
#                         st.subheader("⬇️")
#                         st.subheader(f":blue[{prediction[0]}]")
#                 else:
#                     st.warning(message)

#         else:
#             st.warning("No Data found, please upload your data!")
#             st.page_link("pages/main.py",icon='🏠')
# else :#os.path.exists(model_path)
#     st.warning("No Model found, please train the models!")
#     st.page_link("pages/ML_Pipeline.py",icon='👨🏻‍🌾')

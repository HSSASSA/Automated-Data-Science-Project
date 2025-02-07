import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

global map 
map = { 'col': {'male':1, "female":0}}

class DataTransformation:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.imputer = SimpleImputer(strategy='mean')

    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
        try:
            df_copy = df.copy()
            if strategy == 'drop':
                df_copy.dropna(inplace=True)
            else:
                for column in df_copy.columns:
                    if df_copy[column].isnull().sum() > 0:
                        df_copy[column] = self.imputer.fit_transform(df_copy[[column]])
            return df_copy
        except Exception as e:
            print('Error' + str(e) + 'in handling missing values')
            raise

    def encode_categorical_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df_copy = df.copy()
            categorical_cols = df_copy.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                df_copy[col] = self.label_encoder.fit_transform(df_copy[col])

            return df_copy
        except Exception as e:
            print('Error' + str(e) + 'in encoding categorical columns')
            raise

    def scale_numerical_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df_copy = df.copy()
            numerical_cols = df_copy.select_dtypes(include=['int64', 'float64']).columns
            df_copy[numerical_cols] = self.scaler.fit_transform(df_copy[numerical_cols])
            return df_copy
        except Exception as e:
            print('Error' + str(e) + 'in scaling numerical columns')
            raise

    def transform_data(self, df: pd.DataFrame, missing_value_strategy: str = 'mean') -> pd.DataFrame:
        df = self.handle_missing_values(df, missing_value_strategy)
        df = self.encode_categorical_columns(df)
        df = self.scale_numerical_columns(df)
        return df

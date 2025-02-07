# FILE: /streamlit-app/streamlit-app/src/types/index.py
# This file defines custom types and interfaces used throughout the application, ensuring type safety and clarity in data handling.

from typing import Any, Dict, List, Tuple
import pandas as pd

# Define a type for the DataFrame used in the application
DataFrameType = pd.DataFrame

# Define a type for the transformation parameters
TransformationParams = Dict[str, Any]

# Define a type for model evaluation results
ModelEvaluationResult = Tuple[str, float, float]

# Define a type for the uploaded file
UploadedFileType = 'csv'

# Define a type for the categorical columns
CategoricalColumnsType = List[str]

# Define a type for the numerical columns
NumericalColumnsType = List[str]
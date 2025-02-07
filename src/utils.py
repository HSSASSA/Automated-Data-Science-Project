import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

def visualize_data(df: pd.DataFrame, target_column: str = None):
    """
    Visualize the distribution of features in the DataFrame.
    
    Args:
        df: Input DataFrame
        target_column: Optional; the target column for color coding in plots
    """
    try:
        for column in df.columns:
            plt.figure(figsize=(10, 6))
            if target_column and target_column in df.columns:
                sns.histplot(data=df, x=column, hue=target_column, kde=True, bins=30)
            else:
                sns.histplot(data=df, x=column, kde=True, bins=30)
            plt.title(f'Distribution of {column}')
            plt.show()
    except Exception as e:
        logging.error(f"Error visualizing data: {e}")

def log_dataframe_info(df: pd.DataFrame):
    """
    Log basic information about the DataFrame.
    
    Args:
        df: Input DataFrame
    """
    logging.info(f"DataFrame shape: {df.shape}")
    logging.info(f"DataFrame columns: {df.columns.tolist()}")
    logging.info(f"DataFrame info: {df.info()}")
    logging.info(f"DataFrame head:\n{df.head()}")
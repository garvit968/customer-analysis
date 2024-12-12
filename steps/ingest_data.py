import logging
import pandas as pd
from zenml import step

class IngestData():
    """
    Ingesting data
    """
    def __init__(self, data_path:str):
        self.data_path = data_path
    
    def get_data(self):
        logging.info(f"Ingestion data from {self.data_path}")
        return pd.read_csv(self.data_path)
    
@step
def ingest_data(data_path: str) -> pd.DataFrame:
    """
    Ingesting data from csv

    Args:
    data_path (str): path to the csv file

    Returns:
    pd.DataFrame: data in a pandas DataFrame
    """
    try:
        ingest_data = IngestData(data_path)
        return ingest_data.get_data()
    except Exception as e:
        logging.error(f"Error {e} occured")
        raise e
    
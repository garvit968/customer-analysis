import logging
import pandas as pd
from zenml import step
from typing_extensions import Annotated
from typing import Tuple    
from src.data_cleaning import DataCleaning, DataDivideStrategy, DataPreprocessStrategy

@step
def clean_data(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    try:
        process_strategy = DataPreprocessStrategy()
        data_cleaning = DataCleaning(df, process_strategy)
        processed_data = data_cleaning.handle_data()

        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(processed_data, divide_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()
        logging.info("Data Ingestion Complete")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(e)
        raise e
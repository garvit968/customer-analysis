import logging
import pandas as pd
from zenml import step
from src.model_dev import LinearReg
from sklearn.base import RegressorMixin
from .config import ModelNameConfig

@step
def model_train(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame, config: ModelNameConfig) -> RegressorMixin:
    model = None
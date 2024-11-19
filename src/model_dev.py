import logging
from abc import abstractmethod, ABC
from sklearn.linear_model import LinearRegression

class Model(ABC):
    """
    Abstract for all methods
    """
    @abstractmethod
    def train(self, X_train, y_train):
        pass

class LinearReg(Model):
    try:
        def train(self, X_train, y_train, **kwargs):
            reg = LinearRegression(**kwargs)
            reg.fit(X_train, y_train)
            return reg
    except Exception as e:
        logging.error(f"Error in training model: {e}")
        raise e
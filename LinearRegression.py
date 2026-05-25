import pandas as pd
import numpy as np
class LinearRegression:
    def __init__(self):
        self.slope: np.ndarray | None = None
        self.intercept: float | None = None
    def fit(self, train: pd.DataFrame, target: pd.DataFrame, learning_rate = 0.01, iterations = 1000):
        """
        I am assuming that the data will be all numerical, not categorical.
        It will be stored in the train dataframe, and target dataframe.
        'train' will be containing the training data without the target column.
        'target' will be containing the target column.
        We are using MSE as the cost function.
        We'll be using gradient descent for loss optimization.
        """
        X = train.values
        y = target.values.reshape(-1,1)

        m, n = X.shape

        slope = np.zeros((n,1))
        intercept = 0.0

        for i in range(iterations):
            y_pred = np.dot(X, slope) + intercept

            error = y_pred - y
            mse = np.mean(error**2)
            slope_gradient = (2/m) * np.dot(X.T, error)
            intercept_gradient = (2/m) * np.sum(error)

            slope -= learning_rate * slope_gradient
            intercept -= learning_rate * intercept_gradient

        self.slope = slope
        self.intercept = intercept

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        X_test = test.values
        if self.slope is None or self.intercept is None:
            raise ValueError("Model has not been trained yet.")
        return np.dot(X_test, self.slope) + self.intercept
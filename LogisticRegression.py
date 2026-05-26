import pandas as pd
import numpy as np

class LogisticRegression:
    def __init__(self):
        self.slope: np.ndarray | None = None
        self.intercept: float | None = None

    def fit(self, train: pd.DataFrame, test: pd.DataFrame, learning_rate = 0.01, iterations = 1000):
        """
        I am assuming that the data will be all numerical, not categorical.
        It will be stored in the train dataframe, and target dataframe.
        'train' will be containing the training data without the target column.
        'target' will be containing the target column.
        We are using Log loss as the cost function.
        We'll be using gradient descent for loss optimization.
        """
        X = train.values
        y = test.values.reshape(-1,1)

        n, m = X.shape

        slope = np.zeros((m,1))
        intercept = 0.0

        for i in range(iterations):
            x = np.dot(X, slope) + intercept
            y_pred = 1 / (1 + np.exp(-x))

            error = y - y_pred

            slope_gradient = (1/n) * np.dot(X.T, error)
            intercept_gradient = (1/n) * np.sum(error)

            slope -= learning_rate * slope_gradient
            intercept -= learning_rate * intercept_gradient


        self.slope = slope
        self.intercept = intercept

    def predict(self, test: pd.DataFrame):
        X_test = test.values

        if self.slope is None or self.intercept is None:
            raise ValueError("Model has not been trained yet.")

        z = np.dot(X_test, self.slope) + self.intercept
        probs = 1 / (1 + np.exp(-z))
        predictions = np.where(probs >= 0.5, 1, 0)

        return predictions
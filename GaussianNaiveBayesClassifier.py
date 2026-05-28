import numpy as np
import pandas as pd

class GaussianNaiveBayesClassifier:
    def __init__(self):
        self.classes = None
        self.class_probabilities = {}
        self.class_counts = {}
        self.mean = {}
        self.variance = {}

    def fit(self, train: pd.DataFrame, target: pd.Series):
        """
        train -> feature dataframe
        target -> target column
        Assumes all the features have numerical values 
        """

        X = train
        y = target
        self.classes = y.unique()
        for c in self.classes:
            self.mean[c] = {}
            self.variance[c] = {}
            X_c = X[y == c]
            self.class_counts[c] = len(X_c)
            self.class_probabilities[c] = len(X_c)/len(X)
            for column in X.columns:
                self.mean[c][column] = X_c[column].mean()
                self.variance[c][column] = (X_c[column].var() + 1e-9)

    def gaussianPDF(self, x, mean, variance):
        return (1/np.sqrt(2 * np.pi * variance)) * np.exp(-((x - mean) ** 2) / (2 * variance))
        
    
    def predict(self, test: pd.DataFrame):
        if self.classes is None:
            raise ValueError("Model has not been trained yet.")
        predictions = []
        for _, row in test.iterrows():
            class_scores = {}
            for c in self.classes:
                score = np.log(self.class_probabilities[c])
                for column in test.columns:
                    value = row[column]
                    probability = self.gaussianPDF(value, self.mean[c][column], self.variance[c][column])
                    score += np.log(probability + 1e-9)
                class_scores[c] = score
            predicted_class = max(class_scores, key=class_scores.get)
            predictions.append(predicted_class)

        return predictions
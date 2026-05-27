import pandas as pd
import numpy as np


class NaiveBayesClassifier:

    def __init__(self):
        self.class_probabilities = {}
        self.feature_probabilities = {}
        self.feature_unique_counts = {}
        self.class_counts = {}
        self.classes = None

    def fit(self, train: pd.DataFrame, target: pd.Series):
        """
        train  -> feature dataframe
        target -> target column (Series)
        Assumes all features are categorical.
        """
        X = train
        y = target
        self.classes = y.unique()
        for column in X.columns:
            self.feature_unique_counts[column] = len(X[column].unique())
        for c in self.classes:
            self.feature_probabilities[c] = {}
            X_c = X[y == c]
            self.class_counts[c] = len(X_c)
            self.class_probabilities[c] = len(X_c) / len(X)
            for column in X.columns:
                self.feature_probabilities[c][column] = {}
                for value in X[column].unique():
                    count = len(X_c[X_c[column] == value])
                    self.feature_probabilities[c][column][value] = (count + 1) / (len(X_c) + self.feature_unique_counts[column])

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
                    probability = self.feature_probabilities[c][column].get(value, 1 / (self.class_counts[c] + self.feature_unique_counts[column]))
                    score += np.log(probability)
                class_scores[c] = score
            predicted_class = max(class_scores, key=class_scores.get)
            predictions.append(predicted_class)

        return predictions
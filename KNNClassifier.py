import pandas as pd
import numpy as np
from collections import Counter

class KNNClassifier:
    def euclidean_distance(self, point1, point2):
        return np.sqrt(np.sum((np.array(point1)- np.array(point2))**2))

    def predict(self, train: pd.DataFrame, target: pd.Series, test: pd.DataFrame, k: int):
        """
        train -> feature dataframe
        target -> target column
        test -> test feature dataframe
        k -> number of nearest neighbors to consider for prediction

        I am using Euclidean Distance for calculating the distance
        """
        
        predictions = []
        X_train = train.values
        y_train = target.values
        X_test = test.values

        for test_point in X_test:
            distances = []
            for i in range(len(X_train)):
                dist = self.euclidean_distance(test_point, X_train[i])
                distances.append((dist, y_train[i]))
            
            distances.sort(key=lambda x: x[0])
            k_nearest_labels = [label for _, label in distances[:k]]

            prediction = Counter(k_nearest_labels).most_common(1)[0][0]
            predictions.append(prediction)
        return np.array(predictions)

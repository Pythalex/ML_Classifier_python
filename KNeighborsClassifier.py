"""
Self implementation of the K nearest neighbors classification
"""

import numpy as np

class KNeighborsClassifier:

    def __init__(self):
        pass

    def fit(self, X_train, y_train):
        """
        Memorizes every input data for future prediction
        """
        
        self.dejavu = []
        # Add every X in the memory
        for x, y in zip(X_train, y_train):
            self.dejavu.append((x, y))

    def compute_distance(self, x1, x2):
        """
        Returns the euclidian distance between x1 and x2
        """
        return np.linalg.norm(x1 - x2)

    def classify(self, x):
        """
        Returns the predicted class for x
        """
        miny, mindist = -1, float("infinity")
        for (x_i, y_i) in self.dejavu:
            d = self.compute_distance(x, x_i)
            if d < mindist:
                #print("x = {} : Nouveau NN à {} {} à distance {}".format(x, x_i, y_i, d))
                mindist = d
                miny = y_i
        self.dejavu.append((x, miny))
        return miny
            
    def predict(self, X):
        """
        Returns the vector y such that y_i is the model label for x_i in X
        """
        y_model = np.zeros(len(X))
        for i, x in enumerate(X):
            y_model[i] = self.classify(x)
        return y_model
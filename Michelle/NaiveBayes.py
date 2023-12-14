
import numpy as np


class NaivesBayes:
    def __init__(self, continous=None):
        self.continous = continous

    def fit(self, X, Y):
        self.classes, self.P = np.unique(Y, return_counts=True)
        self.P = self.P/X.shape[0] #Normierung durch die Teilung der Anzahl der einzelnen Werte durch die Länge von Y -> Wahrscheinlichkeitsverteilung

        if self.continous is None:
            self.continous = np.zeros(x.shape[1], dtype=bool)
        Xd = X[:, ~self.continous]
        Xc = X[:, self.continous]

        self.noOfFeaturesC = Xc.shape[1]
        self.noOfFeaturesD = Xd.shaoe[1]

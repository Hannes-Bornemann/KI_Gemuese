
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split



class NaivesBayes:
    def __init__(self, continous=None):
        self.continous = continous

    def fit(self, X, Y):
        self.classes, self.P = np.unique(Y, return_counts=True)
        self.P = self.P/X.shape[0] #Normierung durch die Teilung der Anzahl der einzelnen Werte durch die Länge von Y -> Wahrscheinlichkeitsverteilung

        if self.continous is None:
            self.continous = np.zeros(X.shape[1], dtype=bool)
        Xd = X[:, ~self.continous]
        Xc = X[:, self.continous]

        self.noOfFeaturesC = Xc.shape[1]
        self.noOfFeaturesD = Xd.shape[1]

        fCMax = 0
        self.featurecategories = []
        for i in range(self.noOfFeaturesD):
            self.featurecategories.append(np.unique(Xd[:, i]))
            fCMax = max(fCMax, len(self.featurecategories[i]))

        self.PP = np.zeros((len(self.classes), self.noOfFeaturesD, fCMax))
        for k in range(self.noOfFeaturesD):
            for i, c in enumerate(self.classes):
                for j, f in enumerate(self.featurecategories[k]):
                    xk = (Xd[:, k] == f)
                    theClass = (Y == c)
                    self.PP[i, k, j] = np.sum(xk & theClass) / (np.sum(theClass))

        self.mu = np.zeros((len(self.classes), self.noOfFeaturesC))
        self.sigma = np.zeros((len(self.classes), self.noOfFeaturesC))
        for k in range(self.noOfFeaturesC):
            for i, c in enumerate(self.classes):
                self.mu[i, k] = np.mean(Xc[Y==c, k])
                self.sigma[i, k] = np.std(Xc[Y==c, k])

    def GaussDistribution(self, x, mu, sigma):
        y = np.exp(-0.5*((x-mu)/sigma)**2) / (sigma*np.sqrt(2*np.pi))
        return y
    
    def predictProba(self, X):
        if len(X.shape) == 1: X = X[np.newaxis, :]
        Xd = X[:, ~self.continous]
        Xc = X[:, self.continous]

        #Product k=1:m P(x^(k)|i) * P(i)
        Product = np.ones((X.shape[0], len(self.classes)))
        for i, c in enumerate(self.classes):
            for k in range(self.noOfFeaturesD):
                indK = np.searchsorted(self.featurecategories[k], Xd[:, k])
                Product[:, i] *= self.PP[i, k, indK]
            for k in range(self.noOfFeaturesC):
                Product[:, i] *= self.GaussDistribution(Xc[:, k], self.mu[i, k], self.sigma[i, k])

            Denominator = Product @ self.P #Matrixvektormultiplikation
            PofClass = self.P * Product / Denominator[:, np.newaxis]
            return PofClass
        
    def predict(self, X):
        chosenClass = np.argmax(self.predictProba(X), axis=1)
        return self.classes[chosenClass]
    



# Get the data
col_names = ['contour number', 'aspect ratio', 'extent', 'Blue', 'Green', 'Red', 'Hue']
data = pd.read_csv("output.csv", skiprows=1, header=None, names=col_names)

X = data.iloc[:, :-1].values  # erstes ":" für alle Zeilen, nach dem Komma ":-1" = "alle Spalten bis auf die letzte", Bildung np-Array durch ".values"
Y = data.iloc[:, -1].values.reshape(-1, 1)  # nur letzte Spalte. "reshape" konvertiert Vektor in Spaltenvektor mit 1 Spalte. durch "-1" werden Zeilen automatisch berechnet

# Aufteilen der Daten in Trainings-und Testdaten
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=41)

# #Initialisieren und Trainieren des Klassifikators
# classes = np.unique(Y_train)
# nb_classifier = NaivesBayes(classes)
# nb_classifier.train(X_train, Y_train)

# #Vorhersagen für die Testdaten
# predictions = nb_classifier.predict(X_test)

# #Ausgabe der Vorhersagen
# print("Predictions:", predictions)

nb_classifier = NaivesBayes(continous=False)
nb_classifier.fit(X_train, Y_train)

yP = nb_classifier.predict(X_test)
print(nb_classifier.predictProba(X_test))
incorrect = np.sum(yP != Y_test)
correct = yP.shape[0] -incorrect







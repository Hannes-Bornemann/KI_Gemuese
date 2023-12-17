import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split


import numpy as np

class NaiveBayesClassifier:
    def __init__(self, classes):
        self.classes = classes
        self.class_probs = {}  # P(C)
        self.feature_probs = {}  # P(F_i|C) für diskrete Merkmale
        self.continuous_probs = {}  # Verteilung für kontinuierliche Merkmale

    def train(self, X_train, y_train):
        total_samples = len(y_train)

        for c in self.classes:
            samples_in_class = np.sum(y_train == c)
            self.class_probs[c] = samples_in_class / total_samples

            for i in range(X_train.shape[1]):
                feature_values = X_train[y_train == c, i]

                # Überprüfen, ob das Merkmal diskret oder kontinuierlich ist
                if isinstance(feature_values[0], (int, float)):
                    # Kontinuierliches Merkmal
                    mean, std = np.mean(feature_values), np.std(feature_values)
                    self.continuous_probs[(i, c)] = (mean, std)
                else:
                    # Diskretes Merkmal
                    unique_values, counts = np.unique(feature_values, return_counts=True)
                    prob_dict = dict(zip(unique_values, counts / samples_in_class))
                    if i not in self.feature_probs:
                        self.feature_probs[i] = {}
                    self.feature_probs[i][c] = prob_dict

    def predict(self, X_test):
        predictions = []

        for sample in X_test:
            max_prob = -1
            predicted_class = None

            for c in self.classes:
                prob_c_given_f = np.log(self.class_probs[c])

                for i, feature_value in enumerate(sample):
                    if isinstance(feature_value, (int, float)):
                        # Kontinuierliches Merkmal
                        mean, std = self.continuous_probs.get((i, c), (0, 1e-10))
                        exponent = -(feature_value - mean)**2 / (2 * std**2)
                        prob_c_given_f += np.log(1 / (np.sqrt(2 * np.pi) * std) * np.exp(exponent))
                    elif i in self.feature_probs and c in self.feature_probs[i]:
                        # Diskretes Merkmal
                        prob_c_given_f += np.log(self.feature_probs[i][c].get(feature_value, 1e-10))

                if prob_c_given_f > max_prob or predicted_class is None:
                    max_prob = prob_c_given_f
                    predicted_class = c

            predictions.append(predicted_class)

        return np.array(predictions)

    
# # Annahme: X_train und y_train sind Ihre Trainingsdaten
# X_train = np.array([[1, 'S'], [2, 'M'], [2, 'M'], [1, 'S'], [1, 'S'], [2, 'S'], [2, 'L'], [2, 'L'], [3, 'L']])
# y_train = np.array(['N', 'N', 'Y', 'Y', 'Y', 'N', 'Y', 'N', 'Y'])

# # Annahme: X_test sind Ihre Testdaten
# X_test = np.array([[1, 'M'], [2, 'S'], [3, 'L']])

# # Initialisieren und trainieren Sie den Klassifikator
# classes = np.unique(y_train)
# nb_classifier = NaiveBayesClassifier(classes)
# nb_classifier.train(X_train, y_train)

# # Vorhersagen für die Testdaten machen
# predictions = nb_classifier.predict(X_test)

# # Ausgabe der Vorhersagen
# print("Predictions:", predictions)

# Get the data
col_names = ['contour number', 'aspect ratio', 'extent', 'Blue', 'Green', 'Red', 'Hue']
data = pd.read_csv("output.csv", skiprows=1, header=None, names=col_names)

X = data.iloc[:, :-1].values  # erstes ":" für alle Zeilen, nach dem Komma ":-1" = "alle Spalten bis auf die letzte", Bildung np-Array durch ".values"
Y = data.iloc[:, -1].values.reshape(-1, 1)  # nur letzte Spalte. "reshape" konvertiert Vektor in Spaltenvektor mit 1 Spalte. durch "-1" werden Zeilen automatisch berechnet

# Aufteilen der Daten in Trainings-und Testdaten
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=41)

#Initialisieren und Trainieren des Klassifikators
classes = np.unique(Y_train)
nb_classifier = NaiveBayesClassifier(classes)
nb_classifier.train(X_train, Y_train)

#Vorhersagen für die Testdaten
predictions = nb_classifier.predict(X_test)

#Ausgabe der Vorhersagen
print("Predictions:", predictions)

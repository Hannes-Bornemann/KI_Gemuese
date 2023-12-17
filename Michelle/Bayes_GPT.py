import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split



import numpy as np

class NaiveBayesClassifier:
    def __init__(self, classes):
        self.classes = classes
        self.class_probs = {}  # P(C)
        self.feature_params = {}  # Mittelwert (mean) und Standardabweichung (std) der Merkmale für jede Klasse

    def train(self, X_train, y_train):
        total_samples = len(y_train)

        for c in self.classes:
            samples_in_class = np.sum(y_train == c)
            self.class_probs[c] = samples_in_class / total_samples

            class_samples = X_train[y_train == c, :]

            # Berechne den Mittelwert und die Standardabweichung für jedes Merkmal und jede Klasse
            mean, std = np.mean(class_samples, axis=0), np.std(class_samples, axis=0)
            self.feature_params[c] = {'mean': mean, 'std': std}

    def predict(self, X_test):
        predictions = []

        for sample in X_test:
            max_prob = -np.inf
            predicted_class = None

            for c in self.classes:
                prob_c_given_f = np.log(self.class_probs[c])

                # Verwende die Wahrscheinlichkeitsdichte der Normalverteilung für kontinuierliche Merkmale
                for i, feature_value in enumerate(sample):
                    mean, std = self.feature_params[c]['mean'][i], self.feature_params[c]['std'][i]
                    exponent = -(feature_value - mean)**2 / (2 * std**2)
                    prob_c_given_f += np.log(1 / (np.sqrt(2 * np.pi) * std) * np.exp(exponent))

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

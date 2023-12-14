import numpy as np

class NaiveBayesClassifier:
    def __init__(self, classes):
        self.classes = classes
        self.class_probs = {}  # P(C)
        self.feature_probs = {}  # P(F_i|C)

    def train(self, X_train, y_train):
        # Berechne P(C) für jede Klasse
        total_samples = len(y_train)
        for c in self.classes:
            samples_in_class = np.sum(y_train == c)
            self.class_probs[c] = samples_in_class / total_samples

            # Berechne P(F_i|C) für jedes Merkmal und Klasse
            for i in range(X_train.shape[1]):
                feature_values = X_train[y_train == c, i]
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

            # Berechne P(C|F) für jede Klasse
            for c in self.classes:
                prob_c_given_f = np.log(self.class_probs[c])
                for i, feature_value in enumerate(sample):
                    if i in self.feature_probs and c in self.feature_probs[i]:
                        # Verwende den Logarithmus, um numerische Stabilität zu verbessern
                        prob_c_given_f += np.log(self.feature_probs[i][c].get(feature_value, 1e-10))

                # Aktualisiere die Vorhersage, wenn eine höhere Wahrscheinlichkeit gefunden wird
                if prob_c_given_f > max_prob or predicted_class is None:
                    max_prob = prob_c_given_f
                    predicted_class = c

            predictions.append(predicted_class)

        return np.array(predictions)
    
# Annahme: X_train und y_train sind Ihre Trainingsdaten
X_train = np.array([[1, 'S'], [2, 'M'], [2, 'M'], [1, 'S'], [1, 'S'], [2, 'S'], [2, 'L'], [2, 'L'], [3, 'L']])
y_train = np.array(['N', 'N', 'Y', 'Y', 'Y', 'N', 'Y', 'N', 'Y'])

# Annahme: X_test sind Ihre Testdaten
X_test = np.array([[1, 'M'], [2, 'S'], [3, 'L']])

# Initialisieren und trainieren Sie den Klassifikator
classes = np.unique(y_train)
nb_classifier = NaiveBayesClassifier(classes)
nb_classifier.train(X_train, y_train)

# Vorhersagen für die Testdaten machen
predictions = nb_classifier.predict(X_test)

# Ausgabe der Vorhersagen
print("Predictions:", predictions)

#hi

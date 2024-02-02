import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score

# Lese die Daten ein
col_names = ['contour number', 'aspect ratio', 'extent', 'Blue', 'Green', 'Red', 'Hue', 'class']
data = pd.read_csv("output.csv", skiprows=1, header=None, names=col_names)

# Teile die Daten in Trainings- und Testdaten auf
train, test = train_test_split(data, test_size=.2, random_state=41)

def calculate_prior(df, Y):
    classes, counts = np.unique(df[Y], return_counts=True)
    prior = counts / len(df)
    return prior

def calculate_likelihood_gaussian(x, mean, std):
    p_x_given_y = (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-((x - mean) ** 2) / (2 * std ** 2))
    return p_x_given_y

def naive_bayes_gaussian(df, x, Y):
    prior = calculate_prior(df, Y)
    labels = np.unique(df[Y])
    likelihood = np.zeros(len(labels))

    for j, label in enumerate(labels):
        df_class = df[df[Y] == label]
        for i, feat_name in enumerate(df.columns[:-1]):
            mean = df_class[feat_name].mean()
            std = df_class[feat_name].std()
            likelihood[j] *= calculate_likelihood_gaussian(x[i], mean, std)

    post_prob = prior * likelihood
    Y_pred = np.argmax(post_prob)
    return Y_pred

# Teste das Modell
X_test = test.iloc[:, :-1].values
Y_test = test.iloc[:, -1].values

Y_pred = np.apply_along_axis(lambda x: naive_bayes_gaussian(train, x, "class"), 1, X_test)

# Evaluierung
print(confusion_matrix(Y_test, Y_pred))
print(f1_score(Y_test, Y_pred, average='weighted'))

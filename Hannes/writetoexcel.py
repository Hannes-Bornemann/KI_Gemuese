import pandas as pd

# Erzeuge einige Beispielwerte
data = {
    'Name': ['John', 'Jane', 'Bob'],
    'Age': [25, 30, 22],
    'City': ['Berlin', 'New York', 'London']
}

data1 = {
    'Name': ['John', 'Jane', 'Bob'],
    'Age': [25, 30, 22],
    'City': ['Berlin', 'New York', 'London']
}

# Erzeuge ein DataFrame aus den Daten
df = pd.DataFrame(data)

# Schreibe das DataFrame in eine Excel-Datei
df.to_excel('output.xlsx', index=False)

print("Daten wurden erfolgreich in die Excel-Datei geschrieben.")
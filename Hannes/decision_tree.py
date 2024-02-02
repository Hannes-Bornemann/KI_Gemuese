import numpy as np
import pandas as pd

# Get the data
col_names = ['contour number', 'aspect ratio', 'extent', 'Blue', 'Green', 'Red', 'Hue']
data = pd.read_csv("output.csv", skiprows=1, header=None, names=col_names)
# data = pd.read_excel("output.xlsx", skiprows=1, header=None, names=col_names)
print(data.head(10))

# Node class
# Tree class

# Train-Test split
# Fit the model
# Test the model
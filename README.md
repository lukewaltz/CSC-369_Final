## Using K-Nearest Neighbors to Predict whether a Tumor is Benign or Malignant

# Importing Data From Kaggle
Data is small enough to simply commit the data file to the git repo.

# Data Parsing and Normalization
Parsing
- id: First column, kept as a string.
- diagnosis: Converted to 1 or 0 for binary classification.
- features: Remaining columns converted to Double.

Normalizing
- \[ X_{normalized} = \frac{X - X_{min}}{X_{max} - X_{min}} \]

- \( X \): The original value of the feature.
- \( X_{min} \): The minimum value of the feature in the dataset.
- \( X_{max} \): The maximum value of the feature in the dataset.
- \( X_{normalized} \): The normalized value of the feature, scaled to the range [0, 1].

- This ensures all feature values are scaled proportionally and lie within the range [0, 1].


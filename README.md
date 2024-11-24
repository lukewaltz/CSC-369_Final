# Using K-Nearest Neighbors to Predict whether a Tumor is Benign or Malignant

### Importing Data From Kaggle
Data is small enough to simply commit the data file to the git repo.

### Data Parsing and Normalization
Parsing
- id: First column, kept as a string.
- diagnosis: Converted to 1 or 0 for binary classification.
- features: Remaining columns converted to Double.

Normalizing  
- normalized_value = (x - min(x)) / (max(x) - min(x))
- if max(x) == min(x), normalized_value = 0.0



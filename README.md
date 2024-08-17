# Crop Yield Prediction Analysis

## Overview
This document outlines the process of loading, cleaning, and analyzing crop yield data, and subsequently training multiple machine learning models to predict crop yields based on several input features. The data is processed and visualized, and predictions are made using the processed data.

## Data Loading and Preprocessing
The data is loaded from a CSV file named `yield_df.csv`. Initial exploration includes viewing the first few rows, checking for null values, duplicate data, and summarizing the data's descriptive statistics.

### Steps:
1. **Load Data**: Data is loaded using pandas.
2. **Initial Exploration**:
   - Display the first few entries.
   - Drop an unnamed column that is not needed.
   - Check the shape and information about null values.
   - Describe statistical summary.
3. **Data Cleaning**:
   - Drop duplicate entries.
   - Remove rows where 'average_rain_fall_mm_per_year' is not a numeric value.

### Code Snippets:
```python
import pandas as pd

df = pd.read_csv('yield_df.csv')
df.drop('Unnamed: 0', axis=1, inplace=True)
df.drop_duplicates(inplace=True)
```
## Data Visualization
Visualizations are created using seaborn and matplotlib to show the count of entries by area and the distribution of crop items.

## Visualizations:
Count of Entries by Area: A count plot showing the frequency of entries for different areas.
Yield per Item: Bar plot showing total yield per crop item.
Feature Engineering
Convert categorical features into numerical format using one-hot encoding and scale the numeric features.

## Preprocessing Steps:
OneHotEncoder for categorical variables.
StandardScaler for numeric variables.

```python
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(transformers=[
    ('onehotencoder', OneHotEncoder(drop='first'), [4, 5]),
    ('standardization', StandardScaler(), [0, 1, 2, 3])
], remainder='passthrough')
```
## Model Building
Multiple regression models are trained to predict the yield. Models include Linear Regression, Lasso, Ridge, K-Neighbors Regressor, and Decision Tree Regressor.

## Model Training and Evaluation:
Each model is evaluated using mean squared error (MSE) and R-squared (R2) score.
```python
from sklearn.metrics import mean_squared_error, r2_score

models = {
    'Linear Regression': LinearRegression(),
    'Lasso': Lasso(),
    'Ridge': Ridge(),
    'K-Neighbors Regressor': KNeighborsRegressor(),
    'Decision Tree Regressor': DecisionTreeRegressor()
}

for name, model in models.items():
    model.fit(X_train_dummy, y_train)
    y_pred = model.predict(X_test_dummy)
    print(f"{name} MSE: {mean_squared_error(y_test, y_pred)} R2: {r2_score(y_test, y_pred)}")
```

##Prediction Function
A function to make predictions using the Decision Tree model.
```python
def prediction(Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item):
    features = np.array([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]])
    transformed_features = preprocessor.transform(features)
    return dtr.predict(transformed_features)
```
##Saving Models and Preprocessors
The trained Decision Tree model and the preprocessor are saved using pickle for future use.
```python
import pickle

with open('dtr.pkl', 'wb') as f:
    pickle.dump(dtr, f)

with open('preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)
```
##Environment Setup
Print versions of key libraries to ensure compatibility.
```python
import sklearn
print(sklearn.__version__)
import numpy
print(numpy.__version__)
```
##Conclusion
This script effectively handles data loading, cleaning, visualization, model training, and prediction for crop yield based on provided data. It utilizes several advanced machine learning techniques and data processing steps to ensure accurate predictions.

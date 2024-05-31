#lasso regression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import math
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df=pd.read_csv('weather_data_numeric.csv')[:1000].dropna()
vapour_pressure=df['vapor_pressure_deficit (kPa)']
solar_radiation=df['direct_radiation (W/m²)']

X = df[['vapor_pressure_deficit (kPa)', 'direct_radiation (W/m²)']]
y = df['temperature_2m']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Lasso Regression model
lasso_model = Lasso(alpha=0.1)  # You can adjust the alpha value for stronger or weaker regularization

# Fit the model to the training data
lasso_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = lasso_model.predict(X_test)

# Calculate RMSE (Root Mean Squared Error)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print("Root Mean Squared Error (RMSE):", rmse)
mse=mean_squared_error(y_test, y_pred, squared=True)
print("Mean Squared Error (MSE):", mse)

# Get the coefficients of the model
coefficients = lasso_model.coef_
intercept = lasso_model.intercept_

# print("Coefficients:", coefficients)
# print("Intercept:", intercept)
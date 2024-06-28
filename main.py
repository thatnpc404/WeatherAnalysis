#monte carlo simulation 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import math
from sklearn.linear_model import LinearRegression
from statsmodels.graphics.tsaplots import plot_pacf,plot_acf
from statsmodels.tsa.arima.model import ARIMAResults
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


skip_range=range(1,27)
df=pd.read_csv('weather_data_numeric.csv',nrows=469037-27,skiprows=lambda x:x in skip_range)[:]

#parameters considered (Vapour Pressure , Solar Radiation , Day/Night , Wind Direction)

vapour_pressure=df['vapor_pressure_deficit (kPa)']
solar_radiation=df['direct_radiation (W/m²)']
day=df['is_day ()']
wind_dir=df['winddirection_10m (°)']



X = df[['vapor_pressure_deficit (kPa)', 'direct_radiation (W/m²)','is_day ()','winddirection_10m (°)']]
y = df['temperature_2m']

x_train,x_test,y_train,y_test=train_test_split(X,y)

model=LinearRegression()
model.fit(x_train,y_train)
predicted=model.predict(x_test)
coefficients = model.coef_
intercept = model.intercept_

def forecast_model(vapour_pressure, solar_radiation,day_samples,wind_dir_samples):
    return intercept + coefficients[0] * vapour_pressure + coefficients[1] * solar_radiation+coefficients[2]*day_samples+coefficients[3]*wind_dir_samples

num_simulations = len(df)  # Number of simulations

# Generate random samples for vapour pressure and solar radiation using actual data
vapour_pressure_samples = np.random.choice(vapour_pressure, num_simulations)
solar_radiation_samples = np.random.choice(solar_radiation, num_simulations)
day_samples=np.random.choice(day,num_simulations)
wind_dir_samples=np.random.choice(wind_dir,num_simulations)


# Calculate forecasted temperatures for each simulation
forecasted_temperatures = forecast_model(vapour_pressure_samples, solar_radiation_samples,day_samples,wind_dir_samples)

#residuals = forecasted_temperatures - y
actual_observed_values=np.array(y,dtype=float)
mse_ml=mean_squared_error(actual_observed_values,forecasted_temperatures)

print("Monte Carlo MSE :",mse_ml)



#normal plotting with monte carlo values

plt.figure(figsize=(18, 12))
plt.xlabel('Predicted Values')
plt.ylabel('Forecasted Values')
plt.title('Predicted vs Forecasted Values')
plt.grid(True)
plt.scatter(range(len(forecasted_temperatures)),forecasted_temperatures, color='red', alpha=1, label='Forecasted Values')
plt.plot(range(len(actual_observed_values)),actual_observed_values, color='green', alpha=1, label='Actual Values')
plt.tight_layout()
plt.legend(["Forecasted","Actual"],loc="upper right")
plt.show()

monte carlo graph code
plt.hist(forecasted_temperatures, bins=30, edgecolor='black')
plt.title('Monte Carlo Distribution')
plt.xlabel('Temperature (°C)')
plt.ylabel('Frequency')
plt.show()



confidence_level = 0.95
alpha = 1 - confidence_level

lower_bound = np.percentile(forecasted_temperatures, 100 * alpha / 2)
upper_bound = np.percentile(forecasted_temperatures, 100 * (1 - alpha / 2))


#Implementing SARIMA model


train_data=df["temperature_2m"][:-20]
test_data=df["temperature_2m"][-20:]

order=(2,1,3)
seasonal_order=(1,1,1,12)
model = SARIMAX(train_data, order=order,seasonal_order=seasonal_order)
model_fit = model.fit(verbose=0)
forecast= model_fit.forecast(steps=20)
forecast = pd.Series(forecast,index=test_data.index)


mse1 = mean_squared_error(test_data, forecast)

print("SARIMA MSE:", mse1)




#ARIMA Plot
plt.figure(figsize=(15, 6))
plt.plot(train_data, label='Train Data')
plt.plot(test_data, label='Test Data')
plt.plot(forecast, label='ARIMA Forecast',color="red")
plt.legend(loc="upper right")
plt.show()


#Ensembling Monte Carlo(with Linear Regression) and ARIMA

forecast_ml=forecasted_temperatures
forecast_arima=np.array(forecast)

#ensemble predictions

ensemble_predictions=np.mean([forecast_ml[-20:],forecast_arima],axis=0)

#finding mse manually

mse_en=mean_squared_error(test_data,ensemble_predictions)

print(mse_en)



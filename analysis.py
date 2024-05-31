#analyzing the dataset to split the data accordingly

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import math
import datetime
from sklearn.linear_model import LinearRegression
from statsmodels.graphics.tsaplots import plot_pacf,plot_acf
from statsmodels.tsa.arima.model import ARIMAResults
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import seaborn as sns

#i'll be analyzing quarterly temperatures
#data starting from 1998-04

skip_range=range(1,247609-61320-8760-2150)
df=pd.read_csv('weather_data_numeric.csv',nrows=464593-175379-17520,skiprows=lambda x:x in skip_range,low_memory=False,encoding='utf-8').dropna()


df['time']=pd.to_datetime(df['time'])
df.set_index('time',inplace=True)


df["temperature_2m"]=pd.to_numeric(df["temperature_2m"],errors='coerce')
df['vapor_pressure_deficit (kPa)']=pd.to_numeric(df['vapor_pressure_deficit (kPa)'],errors='coerce')
df['direct_radiation (W/m²)']=pd.to_numeric(df['direct_radiation (W/m²)'],errors='coerce')


daily_data_max=df["temperature_2m"].resample("D").max()
daily_data_min=df["temperature_2m"].resample("D").min()

vp_max=df['vapor_pressure_deficit (kPa)'].resample("D").max()
vp_min=df['vapor_pressure_deficit (kPa)'].resample("D").min()

rad_max=df['direct_radiation (W/m²)'].resample("D").max()
rad_min=df['direct_radiation (W/m²)'].resample("D").min()

print(daily_data_max)
quit()


x_max=[[vp_max,rad_max]]
y_max=daily_data_max


#the max model

x_train,x_test,y_train,y_test=train_test_split(x_max,y_max)


model=LinearRegression()
fit=model.fit(x_train,y_train)
predicted=fit.predict(x_test)


plt.plot(range(len(predicted)),y_test,color="blue")
plt.scatter(range(len(predicted)),predicted,color="red")
plt.show()


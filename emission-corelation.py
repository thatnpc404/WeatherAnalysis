#analyzing the dataset to split the data accordingly
#UPDATED VERSION ON COLAB


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

#Don't consider this code , especially the dataset for emission . Refer the colab version for the same !

skip_range=range(1,247609)
df=pd.read_csv('weather_data_numeric.csv',nrows=469038,skiprows=lambda x:x in skip_range,low_memory=False,encoding='utf-8')[:6601]
data=pd.read_csv("emission_main.csv",encoding='utf-8')[:10]




df['time']=pd.to_datetime(df['time'])
df.set_index('time',inplace=True)

data['yyyymm']=pd.to_datetime(data['yyyymm'])
data.set_index('yyyymm',inplace=True)

data["Total"]=pd.to_numeric(data["Total"],errors='coerce')
df["temperature_2m"]=pd.to_numeric(df["temperature_2m"],errors='coerce')

daily_data=df['temperature_2m'].resample('M').mean()



new_index=pd.date_range(start="1998-04-01",end="1999-01-31",freq='M')


merged=pd.DataFrame(
    {
    'date':new_index,
    'temp':daily_data[3:].values,
    'emission':data["Total"].values
    }
)



corr_matrix=merged.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

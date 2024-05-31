import pandas as pd
import matplotlib.pyplot as plt


skip_range=range(1,8761)
df=pd.read_csv('weather_data_numeric.csv',nrows=469038,skiprows=lambda x:x in skip_range,low_memory=False)[:8760]

df['time']=pd.to_datetime(df['time'])
df.set_index('time',inplace=True)

df["temperature_2m"]=pd.to_numeric(df["temperature_2m"],errors='coerce')


daily_data=df['temperature_2m'].resample('D').mean()
plt.figure(figsize=(10, 6))
plt.plot(daily_data.index, daily_data.values, marker='o')
plt.xlabel('Date')
plt.ylabel('Daily Mean Temperature (°C)')
plt.title('Daily Mean Temperature Plot')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
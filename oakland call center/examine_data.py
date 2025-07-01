import numpy as np
import pandas as pd
import locale
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from holt_winters import HoltWinters


locale.setlocale(locale.LC_TIME, 'en_US.UTF-8')



data_path = '/c/DATA/Datos/Service_requests_received_by_the_Oakland_Call_Center__OAK_311__20250423.csv'

data = pd.read_csv(data_path)
data['DATETIMEINIT'] = pd.to_datetime(data['DATETIMEINIT'], format='%m/%d/%Y %I:%M:%S %p')

# focus on ILLDUMP and get daily counts
illdump_data = data[data['REQCATEGORY'] == 'ILLDUMP'].copy()
illdump_data.set_index('DATETIMEINIT', inplace=True)
daily_illdump_counts = illdump_data.resample('D').size().rename('request_count') # .size() counts non-null rows per group

series = daily_illdump_counts[:'2023-02-07']  # before a streak of nans


#%% check REQCATEGORY

plt.figure()
category_counts = data['REQCATEGORY'].value_counts()
category_counts.plot(kind='barh')
plt.gca().invert_yaxis()
plt.tight_layout()


#%% display number of requests per day

# request_dates = data.loc[data['REQCATEGORY'] == 'ILLDUMP', 'DATETIMEINIT'].dt.date
request_dates = data.loc[:, 'DATETIMEINIT'].dt.date
daily_counts = request_dates.value_counts().sort_index()

plt.figure(figsize=(10, 6))
plt.plot(daily_counts.index, daily_counts.values, '.-')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%a %Y-%m-%d'))




#%% Check out statmodels' seasonal_decompose

decomposition_additive = seasonal_decompose(daily_illdump_counts, model='additive', period=7) # Assuming weekly seasonality

fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

decomposition_additive.observed.plot(ax=axes[0], legend=False)
axes[0].set_ylabel('Observed')
axes[0].set_title('Time Series Decomposition (Additive)')

decomposition_additive.trend.plot(ax=axes[1], legend=False)
axes[1].set_ylabel('Trend')

decomposition_additive.seasonal.plot(ax=axes[2], legend=False)
axes[2].set_ylabel('Seasonal')

decomposition_additive.resid.plot(ax=axes[3], legend=False)
axes[3].set_ylabel('Residual')

plt.xlabel('Date')
plt.tight_layout()
plt.show()


#%% Try a seasonal naive model

num_steps = range(1, 22)
sn_mae = np.empty(len(num_steps))

seasonal_period = 7

for i, steps in enumerate(num_steps):
    forecasts = pd.Series(np.nan, index=series.index)
    for t in range(seasonal_period-steps, len(series)-steps):
        t_previous_season = t + steps - ((steps - 1) // seasonal_period + 1) * seasonal_period
        forecasts.iloc[t+steps] = series.iloc[t_previous_season]
        
    # plt.figure()
    # plt.plot(series)
    # plt.plot(forecasts)
    
    sn_mae[i] = ((series - forecasts).abs()).mean()




#%% Holt-Winters triple exponential smoothing

holt_winters = HoltWinters(seasonal_period=7)
series_numeric = series.to_numpy()
holt_winters.fit(series_numeric, alpha=0.05, beta=0.05, gamma=0.05)

forecasts = holt_winters.forecast(steps=1)
forecasts = pd.Series(forecasts, index=series.index)

plt.figure()
plt.plot(series)
plt.plot(forecasts)


#%% Measure accuracy for different steps into the future

num_steps = range(1, 22)
hw_mae = np.empty(len(num_steps))
errors = np.empty((len(num_steps), len(series)))

for i, steps in enumerate(num_steps):
    forecasts = holt_winters.forecast(steps=steps)
    errors[i, :] = series - forecasts
    hw_mae[i] = ((series - forecasts).abs()).mean()


fig, ax = plt.subplots(1, 2)
ax[0].plot(num_steps, sn_mae, label='seasonal naive')
ax[0].plot(num_steps, hw_mae, label='holt-winters')
ax[0].set_ylabel('mean absolute error')
ax[0].set_xlabel('# days ahead')
ax[0].legend()

# error heatmap
top_error = np.nanpercentile(errors, 99)
bottom_error = np.nanpercentile(errors, 1)
num_bins = 20
error_counts = np.zeros((num_bins, len(num_steps)))
for i in range(len(num_steps)):
    error_counts[:, i], bin_edges = np.histogram(errors[i, :], bins=num_bins, range=(bottom_error, top_error)) 
error_counts /= error_counts.sum(axis=0)

ax[1].imshow(error_counts, aspect='auto', origin='lower', extent=(0.5, len(num_steps)+0.5, bin_edges[0], bin_edges[-1]))

percentile = 2.5
ax[1].plot(num_steps, np.nanpercentile(errors, percentile, axis=1), color='k', label=f'{round(100-2*percentile)}% interval')
ax[1].plot(num_steps, np.nanpercentile(errors, 100-percentile, axis=1), color='k')
ax[1].set_ylabel('error')
ax[1].set_xlabel('# days ahead')
ax[1].legend()
plt.tight_layout()


#%% Try out statsmodels' version

model = ExponentialSmoothing(series, trend='add', seasonal='mul', seasonal_periods=7)
fit_model = model.fit()
forecast = fit_model.forecast(steps=21)

plt.figure()
plt.plot(series)
plt.plot(forecast)


#%% Measure accuracy for different steps into the future

num_steps = range(1, 20)
n_steps_forecasts = np.full((len(num_steps), len(series)), np.nan)

for t in range(14, len(series)-steps-1):
    if t % 100 == 0:
        print(f'{t}/{len(series)}')
    model = ExponentialSmoothing(series[:t], trend='add', seasonal='mul', seasonal_periods=7)
    fit_model = model.fit()
    forecasts = fit_model.forecast(steps=max(num_steps))
    for i, steps in enumerate(num_steps):
        n_steps_forecasts[i, t+steps-1] = forecasts.iloc[i]
        
np.save('/home/eloy/Code/ml_playground/oakland call center/results/statmodels_holt_winters_n_steps_forecasts.npy', n_steps_forecasts)


statsm_hw_mae = np.empty(len(num_steps))
for i in range(len(num_steps)):
    statsm_hw_mae[i] = ((series - n_steps_forecasts[i, :]).abs()).mean()
    
    
plt.figure()
plt.plot(num_steps, sn_mae, label='seasonal naive')
plt.plot(num_steps, hw_mae, label='my holt-winters')
plt.plot(num_steps, statsm_hw_mae, label='sm holt-winters')
plt.legend()
plt.ylabel('mean absolute error')
plt.xlabel('# days ahead')


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    







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


# check REQCATEGORY

plt.figure()
category_counts = data['REQCATEGORY'].value_counts()
category_counts.plot(kind='barh')
plt.gca().invert_yaxis()
plt.tight_layout()


# display number of requests per day

# request_dates = data.loc[data['REQCATEGORY'] == 'ILLDUMP', 'DATETIMEINIT'].dt.date
request_dates = data.loc[:, 'DATETIMEINIT'].dt.date
daily_counts = request_dates.value_counts().sort_index()

plt.figure(figsize=(10, 6))
plt.plot(daily_counts.index, daily_counts.values, '.-')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%a %Y-%m-%d'))


# focus on ILLDUMP and get daily counts
illdump_data = data[data['REQCATEGORY'] == 'ILLDUMP'].copy()
illdump_data.set_index('DATETIMEINIT', inplace=True)
daily_illdump_counts = illdump_data.resample('D').size().rename('request_count') # .size() counts non-null rows per group


# decomposition_additive = seasonal_decompose(daily_illdump_counts, model='additive', period=7) # Assuming weekly seasonality

# fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

# decomposition_additive.observed.plot(ax=axes[0], legend=False)
# axes[0].set_ylabel('Observed')
# axes[0].set_title('Time Series Decomposition (Additive)')

# decomposition_additive.trend.plot(ax=axes[1], legend=False)
# axes[1].set_ylabel('Trend')

# decomposition_additive.seasonal.plot(ax=axes[2], legend=False)
# axes[2].set_ylabel('Seasonal')

# decomposition_additive.resid.plot(ax=axes[3], legend=False)
# axes[3].set_ylabel('Residual')

# plt.xlabel('Date')
# plt.tight_layout()
# plt.show()



# set bad period to nan
daily_illdump_counts['2023-02-08':'2023-04-18'] = np.nan


#%% Holt-Winters triple exponential smoothing

holt_winters = HoltWinters(seasonal_period=7)
series = daily_illdump_counts[:'2023-02-07']
holt_winters.fit(series.to_numpy(), alpha=0.05, beta=0.05, gamma=0.05)

forecasts = holt_winters.forecast(steps=1)
forecasts = pd.Series(forecasts, index=series.index)

plt.figure()
plt.plot(series)
plt.plot(forecasts)


#%% Measure accuracy for different steps into the future

num_steps = range(1, 20)
mean_absolute_error = np.empty(len(num_steps))
errors = np.empty((len(num_steps), len(series)))

for i, steps in enumerate(num_steps):
    forecasts = holt_winters.forecast(steps=steps)
    errors[i, :] = series - forecasts
    mean_absolute_error[i] = ((series - forecasts).abs()).mean()


fig, ax = plt.subplots(1, 2)
ax[0].plot(num_steps, mean_absolute_error)
ax[0].set_ylabel('mean absolute error')
ax[0].set_xlabel('# days ahead')

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

percentiles = [50, 25, 5, 2.5]
for percentile_num, percentile in enumerate(percentiles):
    ax[1].plot(num_steps, np.nanpercentile(errors, percentile, axis=1), color=colors[percentile_num], label=f'{percentile}%')
    if percentile != 50:
        ax[1].plot(num_steps, np.nanpercentile(errors, 100-percentile, axis=1), color=colors[percentile_num])
ax[1].set_ylabel('error percentiles')
ax[1].set_xlabel('# days ahead')
ax[1].legend()
plt.tight_layout()


#%% Try out statsmodels' version

series = daily_illdump_counts[:'2023-02-07']
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


mean_absolute_error_statsm = np.empty(len(num_steps))
for i in range(len(num_steps)):
    mean_absolute_error_statsm[i] = ((series - n_steps_forecasts[i, :]).abs()).mean()
    
    
plt.figure()
plt.plot(num_steps, mean_absolute_error, label='mine')
plt.plot(num_steps, mean_absolute_error_statsm, label='statsmodels')
plt.legend()
plt.ylabel('mean absolute error')
plt.xlabel('# days ahead')


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    







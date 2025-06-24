import numpy as np
import pandas as pd
import locale
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.seasonal import seasonal_decompose
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
    errors[i] = series - forecasts
    mean_absolute_error[i] = np.mean(np.abs(series - forecasts))


fig, ax = plt.subplots(1, 2)
ax[0].plot(num_steps, mean_absolute_error)
ax[0].set_ylabel('mean absolute error')
ax[0].set_xlabel('# days ahead')



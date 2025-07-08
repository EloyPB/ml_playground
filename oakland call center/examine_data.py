import numpy as np
import pandas as pd
import locale
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from holt_winters import HoltWinters
from statsmodels.tsa.statespace.sarimax import SARIMAX


locale.setlocale(locale.LC_TIME, 'en_US.UTF-8')



data_path = '/c/DATA/Datos/Service_requests_received_by_the_Oakland_Call_Center__OAK_311__20250423.csv'
results_path = '/home/eloy/Code/ml_playground/oakland call center/results'

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
plt.xlabel('Count')
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
seasonal_period = 7

n_steps_forecasts = np.full((len(num_steps), len(series)), np.nan)
sn_mae = np.empty(len(num_steps))
sn_mape = np.empty(len(num_steps))


for i, steps in enumerate(num_steps):
    lag = ((steps - 1) // seasonal_period + 1) * seasonal_period
    for t in range(lag - steps, len(series) - steps):
        n_steps_forecasts[i, t + steps] = series.iloc[t + steps - lag]

    # plt.figure()
    # plt.plot(series)
    # plt.plot(pd.Series(n_steps_forecasts[i], index=series.index))
    
    sn_mae[i] = ((series - n_steps_forecasts[i]).abs()).mean()
    sn_mape[i] = (((series - n_steps_forecasts[i]).abs())/series*100).mean()


np.save(f'{results_path}/n_steps_forecasts/seasonal_naive.npy', n_steps_forecasts)


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

n_steps_forecasts = np.full((len(num_steps), len(series)), np.nan)
errors = np.empty((len(num_steps), len(series)))
hw_mae = np.empty(len(num_steps))
hw_mape = np.empty(len(num_steps))

for i, steps in enumerate(num_steps):
    n_steps_forecasts[i] = holt_winters.forecast(steps=steps)
    errors[i, :] = series - n_steps_forecasts[i]
    hw_mae[i] = ((series - n_steps_forecasts[i]).abs()).mean()
    hw_mape[i] = (((series - n_steps_forecasts[i]).abs())/series*100).mean()
    
    
np.save(f'{results_path}/n_steps_forecasts/my_holt_winters.npy', n_steps_forecasts)


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


#%% Compute rolling forecasts

num_steps = range(1, 22)
n_steps_forecasts = np.full((len(num_steps), len(series)), np.nan)

for t in range(14, len(series)-min(num_steps)):
    if t % 100 == 0:
        print(f'{t}/{len(series)}')
    model = ExponentialSmoothing(series[:t+1], trend='add', seasonal='mul', seasonal_periods=7)
    fit_model = model.fit()
    forecasts = fit_model.forecast(steps=max(num_steps))
    for i, steps in enumerate(num_steps):
        if t + steps < len(series):
            n_steps_forecasts[i, t+steps] = forecasts.iloc[i]
        
np.save(f'{results_path}/n_steps_forecasts/statmodels_holt_winters.npy', n_steps_forecasts)


#%% Compare performance 

n_steps_forecasts = np.load(f'{results_path}/n_steps_forecasts/statmodels_holt_winters.npy')


statsm_hw_mae = np.empty(len(num_steps))
statsm_hw_mape = np.empty(len(num_steps))
for i in range(len(num_steps)):
    statsm_hw_mae[i] = ((series - n_steps_forecasts[i, :]).abs()).mean()
    statsm_hw_mape[i] = (((series - n_steps_forecasts[i, :]).abs())/series*100).mean()
    
    
plt.figure()
plt.plot(num_steps, sn_mae, label='seasonal naive')
plt.plot(num_steps, hw_mae, label='my holt-winters')
plt.plot(num_steps, statsm_hw_mae, label='sm holt-winters')
plt.legend()
plt.ylabel('mean absolute error')
plt.xlabel('# days ahead')
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))


#%% Try out SARIMA model

# Apply seasonal differencing

diff_series = series.diff(periods=7)

# Plot ACF and PACF to choose parameters

plot_acf(diff_series.iloc[7:], lags=50, alpha=0.05)
plt.title('Autocorrelation Function (ACF)')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.grid(True)
plt.show()

plot_pacf(diff_series.iloc[7:], lags=50, alpha=0.05)
plt.title('Partial Autocorrelation Function (PACF)')
plt.xlabel('Lag')
plt.ylabel('Partial Autocorrelation')
plt.grid(True)
plt.show()
    

# Fit model
    
model = SARIMAX(series, order=(2, 0, 2), seasonal_order=(0, 1, 1, 7))
fitted_model = model.fit(maxiter=100) 


# Check residuals
residuals = fitted_model.resid

plt.figure(figsize=(12, 6))
plt.plot(residuals)
plt.title('SARIMA Model Residuals')
plt.xlabel('Date')
plt.ylabel('Residual Value')
plt.grid(True)
plt.show()

# ACF/PACF of residuals
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
plot_acf(residuals, lags=min(40, len(residuals)//2 - 1), ax=axes[0], title='ACF of Residuals')
plot_pacf(residuals, lags=min(40, len(residuals)//2 - 1), ax=axes[1], title='PACF of Residuals')
plt.tight_layout()
plt.show()


np.mean(np.abs(fitted_model.fittedvalues.iloc[7:] - series.iloc[7:]))


forecast_obj = fitted_model.get_forecast(steps=21)
forecast_mean = forecast_obj.predicted_mean
forecast_conf_int = forecast_obj.conf_int()


# Plot original series, fitted values, and forecast
plt.figure(figsize=(15, 7))
plt.plot(series.index, series, label='Original Series')
plt.plot(fitted_model.fittedvalues.index, fitted_model.fittedvalues, color='red', label='Fitted Values')
plt.plot(forecast_mean.index, forecast_mean, color='green', label='Forecast')
plt.fill_between(forecast_conf_int.index,
                 forecast_conf_int.iloc[:, 0],
                 forecast_conf_int.iloc[:, 1], color='lightgreen', alpha=0.3, label='Confidence Interval')
plt.title('SARIMA Model: Original, Fitted, and Forecast')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()


#%% Try auto-arima

from pmdarima import auto_arima

auto_model = auto_arima(series, seasonal=True, m=7, trace=True, error_action='warn', 
                        suppress_warnings=True, stepwise=True)

auto_model.summary()


#%% Measure accuracy

# split data in train / test and train initial model

series_train = series.iloc[:1500]
series_test = series.iloc[1500:]

model = SARIMAX(series_train, order=(2, 0, 2), seasonal_order=(1, 1, 1, 7))
fitted_model = model.fit(maxiter=100) 

num_steps = range(1, 22)
n_steps_forecasts = np.full((len(num_steps), len(series_test)), np.nan)

for t in range(len(series_test)-min(num_steps)):
    if t % 100 == 0:
        print(f'{t}/{len(series_test)}')
        
    new_data_point = series_test.iloc[[t]]
    
    if (t+1)%250 == 0:
        refit = True
        fit_kwargs = {'maxiter': 100}
    else:
        refit = False
        fit_kwargs = {}
        
    fitted_model = fitted_model.append(new_data_point, refit=refit, fit_kwargs=fit_kwargs)
        
    forecast_obj = fitted_model.get_forecast(steps=max(num_steps))
    predicted_values = forecast_obj.predicted_mean
    
    for i, steps in enumerate(num_steps):
        if t+steps < len(series_test):
            n_steps_forecasts[i, t+steps] = predicted_values.iloc[i]
    
    
    
np.save(f'{results_path}/n_steps_forecasts/sarima_(2_0_2)(1_1_1_7).npy', n_steps_forecasts)


#%% Compare performance 

sn_forecasts = np.load(f'{results_path}/n_steps_forecasts/seasonal_naive.npy')
my_hw_forecasts = np.load(f'{results_path}/n_steps_forecasts/my_holt_winters.npy')
sm_hw_forecasts = np.load(f'{results_path}/n_steps_forecasts/statmodels_holt_winters.npy')
sarima_forecasts = np.load(f'{results_path}/n_steps_forecasts/sarima_(2_0_2)(1_1_1_7).npy')

all_forecasts = (sn_forecasts[:, 1500:], my_hw_forecasts[:, 1500:], sm_hw_forecasts[:, 1500:], sarima_forecasts)
model_names = ('seasonal naive', 'my holt-winters', 'sm holt-winters', 'sarima')


fig, ax = plt.subplots(1, 2, figsize=(10, 5))

for model_forecasts, model_name in zip(all_forecasts, model_names):

    mae = np.empty(len(num_steps))
    mape = np.empty(len(num_steps))
    for i in range(len(num_steps)):
        mae[i] = ((series_test - model_forecasts[i, :]).abs()).mean()
        mape[i] = (((series_test - model_forecasts[i, :]).abs())/series_test*100).mean()
    
    ax[0].plot(num_steps, mae, label=model_name)
    ax[1].plot(num_steps, mape, label=model_name)


ax[0].legend()
ax[0].set_ylabel('mean absolute error')
ax[0].set_xlabel('# days ahead')
ax[0].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

ax[1].legend()
ax[1].set_ylabel('mean absolute percentage error (%)')
ax[1].set_xlabel('# days ahead')
ax[1].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.tight_layout()


    
    

    
    
    
    
    







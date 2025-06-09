import numpy as np
import matplotlib.pyplot as plt


class HoltWinters:
    def __init__(self, seasonal_period: int, model_type="multiplicative"):
        self.seasonal_period = seasonal_period
        self.model_type = model_type
        
        self.levels = None
        self.trends = None
        self.seasonals = None
        
        
    def fit(self, series: np.ndarray, alpha: float, beta: float, gamma: float):
        self.levels = np.empty_like(series)
        self.trends = np.empty_like(series)
        self.seasonals = np.empty_like(series)
        
        
        # initial level: average of the first seasonal period
        self.levels[:self.seasonal_period] = series[:self.seasonal_period].mean()
        
        # initial trend: average difference between corresponding observations across two seasonal periods
        self.trends[:self.seasonal_period] = np.mean(series[self.seasonal_period:2*self.seasonal_period] - series[:self.seasonal_period]) / self.seasonal_period
        
        # initial seasonal components
        self.seasonals[:self.seasonal_period] = series[:self.seasonal_period] / self.levels[0]
        
        
        # smoothing
        for t in range(self.seasonal_period, series.size):
            self.levels[t] = alpha * (series[t] / self.seasonals[t - self.seasonal_period]) + (1 - alpha) * (self.levels[t-1] + self.trends[t-1])
            self.trends[t] = beta * (self.levels[t] - self.levels[t-1]) + (1 - beta) * self.trends[t-1]
            self.seasonals[t] = gamma * series[t] / self.levels[t] + (1 - gamma) * self.seasonals[t- self.seasonal_period] 
            
        
    def forecast(self, steps: int) -> np.ndarray:
        forecasts = np.full_like(self.levels, np.nan)
        for t in range(self.seasonal_period, self.levels.size-steps):
            t_previous_season = t + steps - (steps // self.seasonal_period + 1) * self.seasonal_period
            forecasts[t+steps] = (self.levels[t] + steps*self.trends[t]) * self.seasonals[t_previous_season]
        return forecasts

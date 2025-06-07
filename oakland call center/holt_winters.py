import numpy as np


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
        self.trends[:self.seasonal_period] = np.mean(series[self.seasonal_period:2*self.seasonal_period] - series[:self.seasonal_period])
        
        # initial seasonal components
        self.seasonals[:self.seasonal_period] = series[:self.seasonal_period] / self.levels[0]
        
        
        ...
        
        
    def forecast(self):
        ...
        

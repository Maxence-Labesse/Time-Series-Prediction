"""Time Series basis forecasting functions
- ...
- ...
"""
from Modules.Random_Generating import *
import numpy as np
import matplotlib.pyplot as plt


###############
# forecasting #
###############

# Moving average
def moving_average_forecast(series, window_size):
    """Compute the mean of the last few values
    MA does not anticipate trend or seasonality

    Parameters
    ----------
    series: np array
        time Series
    window_size:
        number of values use for forecasting

    Returns
    -------
    np array
    """
    forecast = []
    for time in range(len(series) - window_size):
        forecast.append(series[time:time + window_size].mean())
    return np.array(forecast)


"""
-------------------------------------------------------------------------------------------------------------
"""


# differencing series
def diff_series(series, order):
    """differencing time series

    Parameters
    ----------
    series: np array
        time series
    order: int (default: 2)
        differencing order

    Returns
    -------
    np array
    """
    return series[order:] - series[:-order]

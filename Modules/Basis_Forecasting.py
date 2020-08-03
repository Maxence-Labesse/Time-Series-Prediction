"""Time Series basis forecasting functions
- ...
- ...
"""
from Modules.Random_Generating import *
import numpy as np
import matplotlib.pyplot as plt

#################
# Generating TS #
#################
# Parameters
time = np.arange(4 * 365 + 1, dtype="float32")
baseline = 10
amplitude = 40
slope = 0.05
noise_level = 5

# Create the series
series = 10 + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)
# Update with noise
series += white_noise(time, noise_level, seed=42)

# Plot
plt.figure(figsize=(10, 6))
plot_series(time, series, label='Generated Time Series')
plt.show()

# Train / Valid split
split_time = 1000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]
plt.figure(figsize=(10, 6))
plot_series(time_train, x_train, label="Train")
plt.show()

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid, label='Valid')
plt.show()


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


# Example
"""
from Modules.Metrics import *

moving_avg = moving_average_forecast(series, 30)[split_time - 30:]

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, moving_avg)
plt.show()

diff_series = diff_series(series, 365)
diff_time = time[365:]
plt.figure(figsize=(10, 6))
plot_series(diff_time, diff_series)
plt.show()

# Moving average on diferenced TS
diff_moving_avg = moving_average_forecast(diff_series, 50)[split_time - 365 - 50:]

plt.figure(figsize=(10, 6))
plot_series(time_valid, diff_series[split_time - 365:])
plot_series(time_valid, diff_moving_avg)
plt.show()

# Add trend and seasonality back
diff_moving_avg_plus_past = series[split_time - 365:-365] + diff_moving_avg

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, diff_moving_avg_plus_past)
plt.show()

# MA on past values
diff_moving_avg_plus_smooth_past = moving_average_forecast(series[split_time - 370:-360], 10) + diff_moving_avg

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, diff_moving_avg_plus_smooth_past)
plt.show()
"""

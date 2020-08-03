import numpy as np
import matplotlib.pyplot as plt
from Modules.Random_Generating import *
from Modules.MA_prediction import *
from Modules.Metrics import *

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
# Forecasting #
###############

# Moving Average
moving_avg = moving_average_forecast(series, 30)[split_time - 30:]
plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid, label='series')
plot_series(time_valid, moving_avg, label="mobile average forecast")
plt.show()

# Differencing Series
diff_series = diff_series(series, 365)
diff_time = time[365:]

# Moving average on diferenced TS
diff_moving_avg = moving_average_forecast(diff_series, 50)[split_time - 365 - 50:]
plt.figure(figsize=(10, 6))
plot_series(time_valid, diff_series[split_time - 365:], label='series')
plot_series(time_valid, diff_moving_avg, label='diff MA')
plt.show()

# Add trend and seasonality back
diff_moving_avg_plus_past = series[split_time - 365:-365] + diff_moving_avg
plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid, label='series')
plot_series(time_valid, diff_moving_avg_plus_past, label='diff MA + past')
plt.show()

# MA on past values
diff_moving_avg_plus_smooth_past = moving_average_forecast(series[split_time - 370:-360], 10) + diff_moving_avg
plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid, label='series')
plot_series(time_valid, diff_moving_avg_plus_smooth_past, label='smooth past + diff MA')
plt.show()

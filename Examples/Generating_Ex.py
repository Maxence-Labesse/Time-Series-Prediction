from Modules.Random_Generating import *

###########
# Example #
###########

# Time values
time = np.arange(4 * 365 + 1)
baseline = 10

# trend
series = trend(time, 0.2)
plt.figure(figsize=(10, 6))
plot_series(time, series, format='-', start=200, end=1000, label='Trend')
plt.show()

# Trend + seasonality
amplitude = 40
slope = 0.05
series = trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)
plt.figure(figsize=(10, 6))
plot_series(time, series, label='Trend + Seasonality')
plt.show()

# Serie + Noise #
noise_level = 5
noise = white_noise(time, noise_level, seed=42)
series += noise
plt.figure(figsize=(10, 6))
plot_series(time, series, label='Trend+Seasonality+Noise')
plt.show()

# Autocorrelation
signal = impulses(time, 10, seed=42)
series = autocorrelation(signal, {1: 0.99})
plot_series(time, series, label='add 1 autoccorelation')
plt.plot(time, signal, "k-")
plt.show()

signal = impulses(time, 10, seed=42)
series = autocorrelation(signal, {1: 0.70, 50: 0.2})
plot_series(time, series, label='add 2 autocorrelations')
plt.plot(time, signal, "k-")
plt.show()

# Impulses
series = impulses(time, 10, 10, seed=42)
plot_series(time, series, label='impulses')
plt.show()

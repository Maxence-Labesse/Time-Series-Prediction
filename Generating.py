"""
Time Series generating and plotting functions:
- plot_series: plot a time series
- trend: Generate a time series trend
- seasonal_pattern
- seasonality
- noise
"""

import numpy as np
import matplotlib.pyplot as plt


# Plot Series
def plot_series(time, series, format="-", start=0, end=None, label=None):
    """plot a time series

    Parameters
    ----------
    time: numpy array
        time values
    series: numpy array
        serie values
    format: char
        values plot format
    start: int
        first value to plot
    end: int
        last value to plot
    label: string
        legend
    """
    plt.plot(time[start:end], series[start:end], format, label=label)
    plt.xlabel("Time")
    plt.ylabel("Value")
    if label:
        plt.legend(fontsize=14)
    plt.grid(True)


# Generate Trend
def trend(time, slope=0):
    """
    Generate a time series trend

    Parameters
    ----------
    time: numpy array
        period of time
    slope: float
        trend slope

    Returns
    -------
    numpy array
    """
    return slope * time


""" Example
# create evenly spaced values within a given interval
time = np.arange(4 * 365 + 1)
baseline = 10
series = trend(time, 0.2)

plt.figure(figsize=(10, 6))
plot_series(time, series, format='-', start=200, end=1000, label='serie')
plt.show()
"""


########################
# Generate Seasonality #
########################
def seasonal_pattern(season_time):
    """Generate an arbitrary pattern for time series seasonality

    Parameters
    ----------
    season_time: float


    Returns
    -------
    Series

    """
    # Un pattern pour les premiers 40% de la saisonnalité, un autre pour la suite
    return np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi),
                    1 / np.exp(3 * season_time))


def seasonality(time, period, amplitude=1, phase=1):
    """Repeats the same pattern at each period"""
    # Phase sert à avoir le même pattern avec un lag par rapport au temps
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)


"""
amplitude = 40
series = seasonality(time, period=365, amplitude=amplitude)

plt.figure(figsize=(10, 6))
plot_series(time, series)
plt.show()
"""

################################
# Generate Trend + Seasonality #
################################
slope = 0.05
series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)

plt.figure(figsize=(10, 6))
plot_series(time, series)
plt.show()


#########
# Noise #
#########
def white_noise(time, noise_level=1, seed=None):
    """ créé un vecteur contenant des valeurs random entre [0,1]*noise_level"""
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level


noise_level = 5
noise = white_noise(time, noise_level, seed=42)

plt.figure(figsize=(10, 6))
plot_series(time, noise)
plt.show()

#################
# Serie + Noise #
#################
series += noise

plt.figure(figsize=(10, 6))
plot_series(time, series)
plt.show()

###############
# Forecasting #
###############
# Split in train and valid at 1000(no test here)
split_time = 1000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]


###################
# Autocorrelation #
###################
def autocorrelation(time, amplitude, seed=None):
    rnd = np.random.RandomState(seed)
    p1 = 0.5
    p2 = -0.1
    ar = rnd.randn(len(time) + 50)
    ar[:50] = 100
    for step in range(50, len(time) + 50):
        ar[step] += p1 * ar[step - 50]
        ar[step] += p2 * ar[step - 33]
    return ar[50:] * amplitude


series = autocorrelation(time, 10, seed=42)
plot_series(time[:200], series[:200])
plt.show()


def autocorrelation(time, amplitude, seed=None):
    rnd = np.random.RandomState(seed)
    p = 0.8
    ar = rnd.randn(len(time) + 1)
    for step in range(1, len(time) + 1):
        ar[step] += p * ar[step - 1]
    return ar[1:] * amplitude


series = autocorrelation(time, 10, seed=42)
plot_series(time[:200], series[:200])
plt.show()

# Add trend
series = autocorrelation(time, 10, seed=42) + trend(time, 2)
plot_series(time[:200], series[:200])
plt.show()
# Add trend and seasonality
series = autocorrelation(time, 10, seed=42) + seasonality(time, period=50, amplitude=150) + trend(time, 2)
plot_series(time[:200], series[:200])
plt.show()

# Disrupting serie
series = autocorrelation(time, 10, seed=42) + seasonality(time, period=50, amplitude=150) + trend(time, 2)
series2 = autocorrelation(time, 5, seed=42) + seasonality(time, period=50, amplitude=2) + trend(time, -1) + 550
series[200:] = series2[200:]
# series += noise(time, 30)
plot_series(time[:300], series[:300])
plt.show()


#####################
# Generate impulses #
#####################
def impulses(time, num_impulses, amplitude=1, seed=None):
    rnd = np.random.RandomState(seed)
    impulse_indices = rnd.randint(len(time), size=num_impulses)
    series = np.zeros(len(time))
    for index in impulse_indices:
        series[index] += rnd.rand() * amplitude
    return series


series = impulses(time, 10, 10, seed=42)
plot_series(time, series)
plt.show()


# Add autocorrelation
def autocorrelation(source, ps):
    ar = source.copy()
    # for each TS value
    for step, value in enumerate(source):
        # for each lag in ps
        for lag, p in ps.items():
            if step - lag > 0:
                # add lag
                ar[step] += p * ar[step - lag]
    return ar


# add 1 autocorrelation
signal = impulses(time, 10, seed=42)
series = autocorrelation(signal, {1: 0.99})
plot_series(time, series)
plt.plot(time, signal, "k-")
plt.show()

# add 2 autocorrelations
signal = impulses(time, 10, seed=42)
series = autocorrelation(signal, {1: 0.70, 50: 0.2})
plot_series(time, series)
plt.plot(time, signal, "k-")
plt.show()

#
series_diff1 = series[1:] - series[:-1]
plot_series(time[1:], series_diff1)

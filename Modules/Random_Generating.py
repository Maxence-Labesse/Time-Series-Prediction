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


"""
-------------------------------------------------------------------------------------------------------------
"""


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


"""
-------------------------------------------------------------------------------------------------------------
"""


# Generate Seasonality
def seasonal_pattern(season_time):
    """Generate an arbitrary pattern for time series seasonality

    Parameters
    ----------
    season_time: numpy array [0,1]
        season time

    Returns
    -------
    numpy array
    """
    # pattern is different for 40% first time values
    return np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi),
                    1 / np.exp(3 * season_time))


def seasonality(time, period, amplitude=1, phase=1):
    """Repeats a pattern over periods of time

    Parameters
    ----------
    time: np.array
        time values
    period: int
        period length
    amplitude: int
        series values amplitude
    phase:
        time values lag

    Returns
    -------
    np array
    """
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)


"""
-------------------------------------------------------------------------------------------------------------
"""


# Noise #
def white_noise(time, noise_level=1.0, seed=None):
    """ Generate random values [0,1]*noise_level

    Parameters
    ----------
    time: np.array
        time values
    noise_level: int
        noise amplitude
    seed: int
        random seed

    Returns
    -------
    np array
    """
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level


"""
-------------------------------------------------------------------------------------------------------------
"""


# Autocorrelation
def autocorrelation(source, ps):
    """Add autocorrelation to time series

    Parameters
    ----------
    source: np array
        time series to add autocorrelation
    ps: dict {lag: factor}
        autocorrelation parameters

    Returns
    -------
    np array
    """
    ar = source.copy()
    # for each TS value
    for step, value in enumerate(source):
        # for each lag in ps
        for lag, p in ps.items():
            if step - lag > 0:
                # add lag
                ar[step] += p * ar[step - lag]
    return ar


"""
-------------------------------------------------------------------------------------------------------------
"""


# Generate impulses
def impulses(time, num_impulses, amplitude=1, seed=None):
    """Creates random impulses (time and amplitude)

    Parameters
    ----------
    time: np array
        time values
    num_impulses: int
        number of impulses
    amplitude: int
        impulses amplitude
    seed: int
        random seed
    Returns
    -------
    np array
    """
    rnd = np.random.RandomState(seed)
    impulse_indices = rnd.randint(len(time), size=num_impulses)
    series = np.zeros(len(time))
    for index in impulse_indices:
        series[index] += rnd.rand() * amplitude
    return series

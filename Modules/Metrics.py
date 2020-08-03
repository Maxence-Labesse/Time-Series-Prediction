import numpy as np


# mean squared error
def mse(x_valid, forecast, verbose=True):
    x_diff = x_valid - forecast
    x_diff = np.square(x_diff)
    if verbose:
        print("Mean Squared Error: " + str(round(x_diff.mean(), 2)))
    return x_diff.mean()


"""
-------------------------------------------------------------------------------------------------------------
"""


# mean absolute error
def mae(x_valid, forecast, verbose=True):
    x_diff = x_valid - forecast
    x_diff = np.absolute(x_diff)
    if verbose:
        print("Mean Absolute Error: " + str(round(x_diff.mean(), 2)))
    return x_diff.mean()

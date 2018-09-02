import time
import numpy as np


def TicTocGenerator():
    """Generator that returns time differences"""
    ti = 0           # initial time
    tf = time.time()  # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf - ti  # returns the time difference


TicToc = TicTocGenerator()


def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print("Elapsed time: %f seconds.\n" % tempTimeInterval)


def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)


def smooth(array, smoothing_horizon=100., initial_value=0.):
    """ Smoothing function (moving average) for plotting """
    smoothed_array = []
    value = initial_value
    b = 1. / smoothing_horizon
    m = 1.
    for x in array:
        m *= 1. - b
        lr = b / (1 - m)
        value += lr * (x - value)
        smoothed_array.append(value)
    return np.array(smoothed_array)

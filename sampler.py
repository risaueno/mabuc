import numpy as np
import pymc3 as pm
# from numpy.linalg import inv
# from scipy.stats import beta
# import matplotlib.pyplot as plt
# import math
# from collections import defaultdict
# import sys
# import os
# import random

from config import Config
config = Config()

#--------------------------------------------#
# Load experimental and observational data   #
#--------------------------------------------#

#data_exp = np.load('data_exp.npy')
data_obs = np.load('data_obs.npy')

E_obs = data_obs[:, 1] / (data_obs[:, 0] + data_obs[:, 1])     # E_obs(Y|X)
#E_exp = data_exp[:, 1] / (data_exp[:, 0] + data_exp[:, 1])     # E_exp(Y|do(X))
# P(X) from observational data, X=intent
p_int = np.sum(data_obs, axis=1) / np.sum(data_obs)

s = np.ones((config.K, config.K))  # successes
f = np.ones((config.K, config.K))  # failures

# Exploration + observation data matrix
s_with_obs = s + np.diag(data_obs[:, 1])
f_with_obs = s + np.diag(data_obs[:, 0])

p_wins = s_with_obs / (s_with_obs + f_with_obs)

print(p_wins)

a = np.random.randint(2, 3, size=4)
b = np.random.randint(1, 4, size=4)
print(a)
print(b)
print(a[np.newaxis].T)
print(b[np.newaxis])
print(np.matmul(a[np.newaxis].T, b[np.newaxis]))

basic_model = pm.Model()
with basic_model:

    # Priors for unknown model parameters
    a = pm.Normal('a', mu=10, sd=10, shape=config.K)
    b = pm.Normal('b', mu=10, sd=10, shape=config.K)

    offset = pm.Normal('offset', mu=0, sd=10)
    #sigma = pm.HalfNormal('sigma', sd=1)
    print(a)
    print(b)

    alpha_ = np.tile(a, (config.K, 1)).T
    beta_ = np.tile(b, (config.K, 1))

    offset_ = np.tile(offset, (config.K, config.K))
    #sigma_ = np.tile(sigma, (config.K, config.K))
    print(alpha_.shape)
    print(beta_.shape)

    row1 = a[0] + b
    row2 = a[1] + b
    row3 = a[2] + b
    row4 = a[3] + b

    # Expected values of outcome
    #mu = sigmoid(alpha_, beta_, offset_)
    mu = pm.math.sigmoid(alpha_ + beta_ + offset)
    print(mu)
    print(mu.shape)
    print(p_wins.shape)

with basic_model:
    # Likelihood (sampling distribution) of observations
    Y_obs = pm.Bernoulli('Y_obs', mu=mu, observed=p_wins)

with basic_model:
    # draw 500 posterior samples
    trace = pm.sample(500)

basic_model = pm.Model()
with basic_model:
    a1 = pm.Normal('a1', mu=0, sd=10)
    a2 = pm.Normal('a2', mu=0, sd=10)
    a3 = pm.Normal('a3', mu=0, sd=10)
    a4 = pm.Normal('a4', mu=0, sd=10)
    b1 = pm.Normal('b1', mu=0, sd=10)
    b2 = pm.Normal('b2', mu=0, sd=10)
    b3 = pm.Normal('b3', mu=0, sd=10)
    b4 = pm.Normal('b4', mu=0, sd=10)

    offset = pm.Normal('offset', mu=0, sd=10)
    sigma = pm.HalfNormal('sigma', sd=1)

    alpha = np.tile([a1, a2, a3, a4], (config.K, 1)).T
    beta = np.tile([b1, b2, b3, b4], (config.K, 1))
    offset_ = np.tile(offset, (config.K, config.K))

    mu = sigmoid(alpha, beta, offset_)
    print(mu)

    Y_obs = pm.Normal('Y_obs', mu=mu, sd=sigma, observed=p_wins)


map_estimate = pm.find_MAP(model=basic_model)
print(map_estimate)

with basic_model:
    # obtain starting values via MAP
    start = pm.find_MAP(model=basic_model)

    # instantiate sampler
    step = pm.NUTS()

    # draw posterior samples
    basic_trace = pm.sample(1000, step=step, start=start)


def sigmoid(a, b, c):
    """
    Numerically stable sigmoid function
    """
    x = a + b + c
    if x >= 0:
        z = np.exp(-x)
        return 1 / (1 + z)
    else:
        z = np.exp(x)
        return z / (1 + z)

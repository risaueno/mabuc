import numpy as np
import pymc3 as pm
# from numpy.linalg import inv
# from scipy.stats import beta
import matplotlib.pyplot as plt
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
f_with_obs = f + np.diag(data_obs[:, 0])

p_wins = s_with_obs / (s_with_obs + f_with_obs)

print(p_wins)

# SAMPLE SOME FIRST

# Instantiate (sample) confounders for all T steps
U_samples = np.random.randint(2, size=(config.N_CONFOUNDERS, config.T))
I_samples = np.array(config.intent(U_samples[0], U_samples[1]))

for t in range(100):
    # I = GetIntent(U_samples[0,t], U_samples[1,t]) #intentEqn
    I = I_samples[t]
    covariateIndex = I

    # Select random action for now
    # EXCLUDE DIAGONAL FOR EXPLORATION
    choices = np.delete(np.arange(config.K), I)
    action = np.random.choice(choices)

    # Probability of success
    win_prob = config.THETA[action, covariateIndex]
    reward = np.random.choice(2, p=[1 - win_prob, win_prob])  # Pull arm

    # # Update collected data matrix
    # self.s[action, self.I] += reward
    # self.f[action, self.I] += 1 - reward

    s_with_obs[action, I] += reward
    f_with_obs[action, I] += 1 - reward
    # Payout based on data only
    p_wins = s_with_obs / (s_with_obs + f_with_obs)

print('\n')
print(p_wins)
print('\n')
print(config.THETA)
# print('\n')
# print(s_with_obs)
# print('\n')
# print(f_with_obs)

basic_model = pm.Model()
with basic_model:

    # Priors for unknown model parameters
    a = pm.Normal('a', mu=0, sd=10, shape=(config.K, 1))
    #a = pm.Normal('a', mu=0, sd=10)
    b = pm.Normal('b', mu=0, sd=10, shape=(1, config.K))

    #a = pm.Normal('a', mu=10, sd=10, shape=(4, 1))
    #b = pm.Normal('b', mu=10, sd=10, shape=(1, 4))

    offset = pm.Normal('offset', mu=0, sd=10)
    #sigma = pm.HalfNormal('sigma', sd=1)

    # alpha_ = np.tile(a, (config.K, 1)).T
    # beta_ = np.tile(b, (config.K, 1))
    # offset_ = np.tile(offset, (config.K, config.K))
    # #sigma_ = np.tile(sigma, (config.K, config.K))
    # print(alpha_.shape)
    # print(beta_.shape)

    # row0 = a[0] + b + offset
    # row1 = a[1] + b + offset
    # row2 = a[2] + b + offset
    # row3 = a[3] + b + offset

    # # Expected values of outcome
    # #mu = sigmoid(alpha_, beta_, offset_)
    # mu0 = sigmoid_stable(row0)
    # mu1 = sigmoid_stable(row1)
    # mu2 = sigmoid_stable(row2)
    # mu3 = sigmoid_stable(row3)

    p = pm.Deterministic('p', sigmoid_stable(a + b + offset))

    # p0 = pm.Deterministic('p0', sigmoid_stable(row0))
    # p1 = pm.Deterministic('p1', sigmoid_stable(row1))
    # p2 = pm.Deterministic('p2', sigmoid_stable(row2))
    # p3 = pm.Deterministic('p3', sigmoid_stable(row3))

    #mu0 = sigmoid_stable(a + b + offset)
    # Test just first row
    #p = pm.Deterministic('p', sigmoid_stable(a + b + offset))

    # Likelihood (sampling distribution) of observations

    L = pm.Bernoulli('L', p=p, observed=p_wins)
    # Y_obs0 = pm.Bernoulli('Y_obs0', p=p0, observed=p_wins[0, :])
    # Y_obs1 = pm.Bernoulli('Y_obs1', p=p1, observed=p_wins[1, :])
    # Y_obs2 = pm.Bernoulli('Y_obs2', p=p2, observed=p_wins[2, :])
    # Y_obs3 = pm.Bernoulli('Y_obs3', p=p3, observed=p_wins[3, :])

    # Test jsut first row
    #Y_obs0 = pm.Bernoulli('Y_obs0', p=p, observed=p_wins[0, :])

with basic_model:

    # draw posterior samples
    # start = pm.find_MAP(model=basic_model)
    #trace = pm.sample(10000)
    trace = pm.sample(1000, nuts_kwargs=dict(target_accept=.90), chains=4)

pm.traceplot(trace)
plt.show()
pm.summary(trace).round(5)

a_post = np.array(np.mean(trace[:500]['a'], axis=0))
b_post = np.array(np.mean(trace[:500]['b'], axis=0))
offset_post = np.mean(trace[:500]['offset'], axis=0)

print(a_post)
print(b_post)
print(offset_post)

added = a_post + b_post + offset_post
print(added)

# a_post = np.array([4.358499, 4.745872, 5.553587, 4.025224])
# b_post = np.array([7.946543, 8.070525, 8.103049, 7.913886])
# offset_post = -10.927278

# a_post = np.tile(a_post, (4, 1)).T
# b_post = np.tile(b_post, (4, 1))
# print(a_post)
# print(b_post)

# ??? Looks nothing like p_wins?
y_post = sigmoid_stable(a_post + b_post + offset_post)
print(y_post)
print('\n')
print(p_wins)

ppc = pm.sample_ppc(trace, samples=500, model=basic_model)
print(ppc['L'].shape)
#print(np.asarray(ppc['a']))
#print(ppc['a'])

# with basic_model:
#     # obtain starting values via MAP
#     start = pm.find_MAP(model=basic_model)
#
#     # instantiate sampler
#     step = pm.NUTS()
#
#     # draw 2000 posterior samples
#     basic_trace = pm.sample(1000, step=step, start=start)

# def sigmoid(a, b, c):
#     """
#     Numerically stable sigmoid function
#     """
#     x = a + b + c
#     if x >= 0:
#         z = np.exp(-x)
#         return 1 / (1 + z)
#     else:
#         z = np.exp(x)
#         return z / (1 + z)


def sigmoid_stable(x):
    """
    Numerically stable sigmoid function
    """
    z = np.exp(x)
    return z / (1 + z)

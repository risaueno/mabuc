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
import theano.tensor as tt

from config import Config
config = Config()


def sigmoid_stable(x):
    """
    Numerically stable sigmoid function
    """
    z = np.exp(x)
    return z / (1 + z)


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

print(p_wins[0, :])
print(config.THETA[0, :])

basic_model = pm.Model()
with basic_model:

    # Priors for unknown model parameters
    #a = pm.Normal('a', mu=0, sd=10, shape=config.K)
    a0 = pm.Normal('a', mu=0.0, tau=0.05, testval=0.0)
    b = pm.Normal('b', mu=0.0, tau=0.05, testval=0.0, shape=config.K)
    offset = pm.Normal('offset', mu=0, sd=10)

    row0 = a0 + b + offset

    # # Expected values of outcome
    # mu0 = sigmoid_stable(row0)
    #p0 = pm.Deterministic('p0', sigmoid_stable(row0))
    p0 = pm.Deterministic('p', 1. / (1. + tt.exp(b + a0 + offset)))

    # Likelihood (sampling distribution) of observations
    Y_obs0 = pm.Bernoulli('Y_obs0', p=p0, observed=p_wins[0, :])

    # Test jsut first row
    #Y_obs0 = pm.Bernoulli('Y_obs0', p=p, observed=p_wins[0, :])

with basic_model:

    # draw posterior samples
    start = pm.find_MAP(model=basic_model)
    #step = pm.Metropolis()
    #trace = pm.sample(1000, step=step)
    #trace = pm.sample(10000)
    trace = pm.sample(1000, nuts_kwargs=dict(target_accept=.90), chains=4, start=start)

pm.traceplot(trace)
plt.show()
pm.summary(trace).round(5)

a_post = np.array(np.mean(trace[:100]['a'], axis=0))
b_post = np.array(np.mean(trace[:100]['b'], axis=0))
offset_post = np.mean(trace[:100]['offset'], axis=0)

print(a_post)
print(b_post)
print(offset_post)

# a_post = np.array([4.358499, 4.745872, 5.553587, 4.025224])
# b_post = np.array([7.946543, 8.070525, 8.103049, 7.913886])
# offset_post = -10.927278

a_post = np.tile(a_post, (4, 1)).T
b_post = np.tile(b_post, (4, 1))
print(a_post)
print(b_post)

y_post = sigmoid_stable(a_post + b_post + offset_post)
y_post = 1. / (1. + np.exp(a_post + b_post + offset_post))
print(y_post)
print(p_wins)

print(trace['a'][-5:])
print(trace['b'][-5:])

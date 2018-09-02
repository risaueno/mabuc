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


def sigmoid(x):
    z = np.exp(x)
    return z / (1 + z)


data = np.array([[0.2, 0.3, 0.5, 0.6],
                 [0.6, 0.2, 0.6, 0.5],
                 [0.5, 0.6, 0.2, 0.3],
                 [0.3, 0.5, 0.6, 0.2]])

basic_model = pm.Model()
with basic_model:

    # Priors for unknown model parameters (9 parameters in total)
    a = pm.Normal('a', mu=0, sd=10, shape=(4, 1))
    b = pm.Normal('b', mu=0, sd=10, shape=(1, 4))
    offset = pm.Normal('offset', mu=0, sd=10)

    # Expected values of outcome (4 x 4)
    p = pm.Deterministic('p', sigmoid(a + b + offset))

    # Likelihood (sampling distribution) of observations (4 x 4)
    L = pm.Bernoulli('L', p=p, observed=data)

with basic_model:
    # Draw posterior samples
    trace = pm.sample(1000)

pm.traceplot(trace)
plt.show()
pm.summary(trace).round(2)

a_post = np.array(np.mean(trace[:500]['a'], axis=0))
b_post = np.array(np.mean(trace[:500]['b'], axis=0))
offset_post = np.mean(trace[:500]['offset'], axis=0)
print(a_post)
print(b_post)
print(offset_post)

# ??? Looks nothing like p_wins?
y_post = sigmoid(a_post + b_post + offset_post)
print(y_post)
print('\n')
print(data)

ppc = pm.sample_ppc(trace, samples=1000, model=basic_model)
print(np.mean(ppc['L'], axis=0).round(3))
print(data)

# [[ 0.01   0.01   0.012  0.011]
#  [ 0.003  0.006  0.007  0.008]
#  [ 0.013  0.011  0.013  0.01 ]
#  [ 0.007  0.007  0.008  0.015]]

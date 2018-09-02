import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm

print('Running on PyMC3 v{}'.format(pm.__version__))

# True parameter values
alpha, sigma = 1, 1
beta = [1, 2.5]

# Size of dataset
size = 100

# Predictor variable
X1 = np.random.randn(size)
X2 = np.random.randn(size) * 0.2

# Simulate outcome variable
Y = alpha + beta[0] * X1 + beta[1] * X2 + np.random.randn(size) * sigma
print(Y.shape)

basic_model = pm.Model()

with basic_model:

    # Priors for unknown model parameters
    alpha = pm.Normal('alpha', mu=0, sd=10)  # (1)
    beta = pm.Normal('beta', mu=0, sd=10, shape=2)  # (2)
    sigma = pm.HalfNormal('sigma', sd=1)  # (1)

    print(alpha)
    print(beta)
    print(sigma.shape)

    # Expected value of outcome
    mu = alpha + beta[0] * X1 + beta[1] * X2
    print(mu)
    print(mu.shape)
    print(Y.shape)

    # Likelihood (sampling distribution) of observations
    Y_obs = pm.Normal('Y_obs', mu=mu, sd=sigma, observed=Y)

map_estimate = pm.find_MAP(model=basic_model)
print(map_estimate)

# Works - compare to ground truth.
# {'alpha': array(0.9377878565802533),
# 'beta': array([ 0.89811599,  2.31351856])
# 'sigma_log__': array(-0.06838953450966123),
# 'sigma': array(0.9338966177535436)}

# SAMPLE NOT WORKING!!!

with basic_model:
    # draw 500 posterior samples
    trace = pm.sample(500)

print(trace)

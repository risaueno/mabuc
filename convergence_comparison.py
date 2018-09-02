import numpy as np
import pymc3 as pm
from sklearn import linear_model

from config import Config
config = Config()

# -------------------------------------------- #
# Load experimental and observational data     #
# -------------------------------------------- #

data_obs = np.load('data_obs.npy')  # Observational data
# data_exp = np.load('data_exp.npy')  # Experimental data (not used here)

E_obs = data_obs[:, 1] / (data_obs[:, 0] + data_obs[:, 1])  # E_obs(Y|X)
# E_exp = data_exp[:, 1] / (data_exp[:, 0] + data_exp[:, 1])  # E_exp(Y|do(X))
p_int = np.sum(data_obs, axis=1) / np.sum(data_obs)  # P(I) from observational data

s = np.ones((config.K, config.K))  # Initialise exploration successes
f = np.ones((config.K, config.K))  # Initialise exploration failures

# Exploration + observation data matrix
s_with_obs = (s + np.diag(data_obs[:, 1])).astype(int)  # Successes
f_with_obs = (f + np.diag(data_obs[:, 0])).astype(int)  # Failures
N_data = (s_with_obs + f_with_obs).astype(int)  # Total

# Observed data success rate (4 x 4)
p_wins = s_with_obs / N_data

# -------------------------------------------- #
# OBTAIN DUMMY EXPLORATION DATA                #
# -------------------------------------------- #
# Instantiate (sample) confounders for all T steps
U_samples = np.random.randint(2, size=(config.N_CONFOUNDERS, config.T))
I_samples = np.array(config.intent(U_samples[0], U_samples[1]))

for t in range(1000):
    # I = GetIntent(U_samples[0,t], U_samples[1,t]) #intentEqn
    I = I_samples[t]
    covariateIndex = I

    # Select random action for this purpose (exclude diagonals)
    choices = np.delete(np.arange(config.K), I)
    action = np.random.choice(choices)

    # Pull arm
    win_prob = config.THETA[action, covariateIndex]
    reward = np.random.choice(2, p=[1 - win_prob, win_prob])

    # Update data matrices
    s_with_obs[action, I] += reward
    f_with_obs[action, I] += 1 - reward
    N_data = s_with_obs + f_with_obs
    p_wins = s_with_obs / N_data  # This is y_observed_data (observational & exploration)

# -------------------------------------------- #
# PREPARE DATA FOR GRADIENT DESCENT            #
# -------------------------------------------- #

# Features (one hot vector of action and intent)
x = np.empty((np.sum(N_data), config.K * 2)).astype(int)
# Labels {0, 1}
y = np.empty(np.sum(N_data)).astype(int)

idx = 0  # counter
for i in range(config.K):
    for j in range(config.K):
        # Create one-hot x vectors and ys for this arm-intent combination
        x_ = np.zeros(config.K * 2).astype(int)
        y_ = np.zeros(N_data[i, j]).astype(int)
        x_[i] = 1
        x_[config.K + j] = 1
        x_ = np.tile(x_, (N_data[i, j], 1))
        y_[:s_with_obs[i, j]] = 1

        # Fill in data matrix
        x[idx:idx + N_data[i, j], :] = x_
        y[idx:idx + N_data[i, j]] = y_

        idx += N_data[i, j]

# -------------------------------------------- #
# LOGISTIC REGRESSION (UNREGULARISED)          #
# -------------------------------------------- #

logistic = linear_model.LogisticRegression(C=10e42)  # Large C = no regularisation
logistic.fit(x, y)

logistic_weights = np.squeeze(logistic.coef_)
c = logistic.intercept_
a = logistic_weights[:4]
b = logistic_weights[4:]
a_ = np.tile(a, (config.K, 1)).T
b_ = np.tile(b, (config.K, 1))
y_out_logistic = config.sigmoid(a_ + b_ + c)


# -------------------------------------------- #
# LOGISTIC REGRESSION (SGD REGULARISED)        #
# -------------------------------------------- #

sgd = linear_model.SGDClassifier(loss='log')
sgd.fit(x, y)

sgd_weights = np.squeeze(sgd.coef_)
c = sgd.intercept_
a = sgd_weights[:4]
b = sgd_weights[4:]
a_ = np.tile(a, (config.K, 1)).T
b_ = np.tile(b, (config.K, 1))
y_out_sgd = config.sigmoid(a_ + b_ + c)


# -------------------------------------------- #
# MCMC                                         #
# -------------------------------------------- #
model = pm.Model()
with model:

    # Priors for unknown model parameters
    a = pm.Normal('a', mu=0, sd=10, shape=(config.K, 1))
    b = pm.Normal('b', mu=0, sd=10, shape=(1, config.K))
    offset = pm.Normal('offset', mu=0, sd=10)

    # p = pm.Deterministic('p', config.sigmoid(a + b + offset))
    p = config.sigmoid(a + b + offset)

    # Likelihood (sampling distribution) of observations
    pm.Binomial('L', n=N_data, p=p, observed=s_with_obs)

# MAP
MAP = pm.find_MAP(model=model)  # Find starting point of MCMC
a_ = np.tile(np.squeeze(MAP['a']), (4, 1)).T
b_ = np.tile(np.squeeze(MAP['b']), (4, 1))
c_ = MAP['offset']
y_MAP = config.sigmoid(a_ + b_ + c_)

# Draw posterior samples
with model:
    # THIS TAKES A WHILE TO RUN!
    trace = pm.sample(config.TRACE_LENGTH, nuts_kwargs=dict(target_accept=.9,
                      max_treedepth=20), chains=config.N_MCMC_CHAINS, start=MAP)

# Posterior from PPC sampled observations
ppc = pm.sample_ppc(trace, samples=500, model=model)
ppc_result = np.mean(ppc['L'], axis=0) / N_data

# Posteiror from trace
a_post = np.array(np.mean(trace[:200]['a'], axis=0))
b_post = np.array(np.mean(trace[:200]['b'], axis=0))
offset_post = np.mean(trace[:200]['offset'], axis=0)
y_post = config.sigmoid(a_post + b_post + offset_post)


# -------------------------------------------- #
# COMPARE WITH ACTUAL DATA                     #
# -------------------------------------------- #
round = 2
print('y_observed = \n {} \n'.format(p_wins.round(round)))
print('y_logistic_unregularised = \n {} \n'.format(y_out_logistic.round(round)))
print('y_logistic_sgd_regularised = \n {} \n'.format(y_out_sgd.round(round)))
print('y_posterior_ppc = \n {} \n'.format(ppc_result.round(round)))
print('y_posterior_trace = \n {} \n'.format(y_post.round(round)))
print('y_MAP = \n {} \n'.format(y_MAP.round(round)))

import numpy as np
import pymc3 as pm
from sklearn import linear_model
import matplotlib.pyplot as plt
from helper import tic, toc, smooth

from config import Config
config = Config()

plot_traceplot = False
run_gradient_descent = False

# -------------------------------------------- #
# Load experimental and observational data     #
# -------------------------------------------- #

data_obs = np.load('data_obs.npy')  # Observational data
data_exp = np.load('data_exp.npy')  # Experimental data (not used here)

E_obs = data_obs[:, 1] / (data_obs[:, 0] + data_obs[:, 1])  # E_obs(Y|X)
E_exp = data_exp[:, 1] / (data_exp[:, 0] + data_exp[:, 1])  # E_exp(Y|do(X))
p_int = np.sum(data_obs, axis=1) / np.sum(data_obs)  # P(I) from observational data

s = np.ones((config.K, config.K))  # Initialise exploration successes
f = np.ones((config.K, config.K))  # Initialise exploration failures

# Exploration + observation data matrix
s_with_obs = (s + np.diag(data_obs[:, 1])).astype(int)  # Successes
f_with_obs = (f + np.diag(data_obs[:, 0])).astype(int)  # Failures
N_data = (s_with_obs + f_with_obs).astype(int)  # Total

# Observed data success rate (4 x 4)
p_wins = s_with_obs / N_data

print(data_exp)
print(E_exp)

# -------------------------------------------- #
# OBTAIN DUMMY EXPLORATION DATA                #
# -------------------------------------------- #
# Instantiate (sample) confounders for all T steps
U_samples = np.random.randint(2, size=(config.N_CONFOUNDERS, config.T))
I_samples = np.array(config.intent(U_samples[0], U_samples[1]))

for t in range(100):
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
    p_wins = s_with_obs / N_data
    # p_wins is y_observed_data (observational & exploration)

if run_gradient_descent:

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
    intercept = logistic.intercept_
    a = logistic_weights[:4]
    b = logistic_weights[4:]
    a_ = np.tile(a, (config.K, 1)).T
    b_ = np.tile(b, (config.K, 1))
    y_out_logistic = config.sigmoid(a_ + b_ + intercept)

    # -------------------------------------------- #
    # LOGISTIC REGRESSION (SGD REGULARISED)        #
    # -------------------------------------------- #

    sgd = linear_model.SGDClassifier(loss='log')
    sgd.fit(x, y)

    sgd_weights = np.squeeze(sgd.coef_)
    intercept = sgd.intercept_
    a = sgd_weights[:4]
    b = sgd_weights[4:]
    a_ = np.tile(a, (config.K, 1)).T
    b_ = np.tile(b, (config.K, 1))
    y_out_sgd = config.sigmoid(a_ + b_ + intercept)


# -------------------------------------------- #
# MCMC                                         #
# -------------------------------------------- #
model = pm.Model()
with model:

    # Priors for unknown model parameters
    #a = pm.Normal('a', mu=0, sd=10, shape=(config.K, 1))
    #b = pm.Normal('b', mu=0, sd=10, shape=(1, config.K))
    #intercept = pm.Normal('intercept', mu=0, sd=10)

    hyper_mu = pm.Normal('hyper_mu', mu=0, sd=10)
    hyper_sd = pm.Gamma('hyper_sd', alpha=0.01, beta=0.001)
    # alpha = 10, beta = 0.001: more like data sd=(alpha/beta^2)
    # wide variance = less information for prior, data is stronger
    # alpha=100, beta=1: QUITE CLOSE to observed data! Also more stable?
    c = pm.Normal('c', mu=hyper_mu, sd=hyper_sd, shape=(config.K, config.K))
    # mask = np.ones((config.K, config.K))
    # np.fill_diagonal(mask, 0)

    # p = pm.Deterministic('p', config.sigmoid(a + b + offset))
    # p = config.sigmoid(a + b + offset)
    #p = config.sigmoid(a + b + c)
    p = config.sigmoid(c)

    # Likelihood (sampling distribution) of observations
    pm.Binomial('L', n=N_data, p=p, observed=s_with_obs)

# MAP
MAP = pm.find_MAP(model=model)  # Find starting point of MCMC
#a_ = np.tile(np.squeeze(MAP['a']), (4, 1)).T
#b_ = np.tile(np.squeeze(MAP['b']), (4, 1))
c_ = np.squeeze(MAP['c'])
#intercept_ = MAP['intercept']
# y_MAP = config.sigmoid(a_ + b_ + c_ + intercept_)
y_MAP = config.sigmoid(c_)

#tic()
# Draw posterior samples
with model:
    # THIS TAKES A WHILE TO RUN!
    trace = pm.sample(1000, nuts_kwargs=dict(target_accept=.9,
                      max_treedepth=20), chains=config.N_MCMC_CHAINS)

#toc()

if plot_traceplot:

pm.traceplot(trace)
plt.show()

# Posterior from PPC sampled observations
ppc = pm.sample_ppc(trace, samples=500, model=model)
ppc_result = np.mean(ppc['L'], axis=0) / N_data

# Posteiror from trace
# a_post = np.array(np.mean(trace[:200]['a'], axis=0))
#b_post = np.array(np.mean(trace[:200]['b'], axis=0))
c_post = np.mean(trace[:100]['c'], axis=0)
# intercept_post = np.mean(trace[:200]['intercept'], axis=0)
#y_post = config.sigmoid(a_post + b_post + c_post)
y_post = config.sigmoid(c_post)


# -------------------------------------------- #
# COMPARE WITH ACTUAL DATA                     #
# -------------------------------------------- #
round = 2
print('y_truth = \n {} \n'.format(config.THETA.round(round)))
print('y_observed = \n {} \n'.format(p_wins.round(round)))
print('y_logistic_unregularised = \n {} \n'.format(y_out_logistic.round(round)))
print('y_logistic_sgd_regularised = \n {} \n'.format(y_out_sgd.round(round)))
print('y_posterior_ppc = \n {} \n'.format(ppc_result.round(round)))
print('y_posterior_trace = \n {} \n'.format(y_post.round(round)))
print('y_MAP = \n {} \n'.format(y_MAP.round(round)))
print('N_data = \n {} \n'.format(N_data))

# Result: C & intercept only - MAP doesn't work
# Why SHOULD they share alpha and beta parameters?
# If a cell doesn't have enough exploration data.
# Where should the information come from? Surely it will be from something like
# the E[Y] constraint, because they share parameters in the system of
# equations. Surely we need to be able to inforporate that.

# IF the hyperprior constrains the cells with LESS EXAMPLES to hover around
# 0.5 then do we not want that?? When z=0 this happens which is taken care of
# by prior ?

# Trying different parameters like (a, b, intercept) , (c, intercept) etc
# Way to evaluate how good they are? --> make them learn bandit

# C only : samplnig is a lot faster, 17 seconds as opposed to 3 mins.

# Parameters (0.1, 0.1)
# How to stop it from exploring so much for observational data when there are so many!?
# --> How to do TS sampling, high variance gets selected more

# Id we give data more power it doesn't explore enough,
# if we have a strong prior it doesn't update the values that reflect the data
# quick enough.
# Having to tune this hyperparameter - not ideal in practice?
# If we incorporate additional information maybe it's better?

# -------- TEST

sample1 = pm.sample_ppc(trace, samples=1, model=model)
sample_result1 = np.mean(sample1['L'], axis=0) / N_data
print(sample_result1.round(2), '\n')

sample2 = pm.sample_ppc(trace, samples=1, model=model)
sample_result2 = np.mean(sample2['L'], axis=0) / N_data
print(sample_result2.round(2), '\n')

sample3 = pm.sample_ppc(trace, samples=1, model=model)
sample_result3 = np.mean(sample3['L'], axis=0) / N_data
print(sample_result3.round(2), '\n')

PProb = np.random.rand((config.T))

plt.figure()

ax = plt.subplot(211)
ax.plot(smooth(PProb), label=config.ALGORITHMS[0])
ax.set_yticks(np.arange(0, 1., 0.1))
ax.set_xticks(np.arange(0, config.T, config.T / 20))
ax.grid()

ax2 = plt.subplot(212)
ax2.plot(smooth(PProb), label=config.ALGORITHMS[0])
#ax2.set_yticks(np.arange(0, 1., 0.1))
ax2.set_xticks(np.arange(0, config.T, config.T / 20))
ax2.grid()

plt.subplot(211)
plt.xlabel('t')
plt.ylabel('Probability of optimal action')
plt.legend()

plt.subplot(212)
plt.xlabel('t')
plt.ylabel('Proportion of reward')
plt.legend()
# plt.legend()

plt.show()

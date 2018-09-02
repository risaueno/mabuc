#%%
import numpy as np
import pymc3 as pm
import theano.tensor as tt
import matplotlib.pyplot as plt
from helper import tic, toc
from theano.printing import pydotprint

from config import Config
config = Config()

plot_traceplot = False
run_gradient_descent = False

#%%
import datetime
print(str(datetime.datetime.now()).split('.')[0])

print(np.arange(config.T) / config.T)

#%%

U_samples_list = np.empty((config.N, config.N_CONFOUNDERS, config.T))
I_samples_list = np.empty((config.N, config.T))

for i in range(config.N):
    U_samples = np.random.randint(2, size=(config.N_CONFOUNDERS, config.T))
    I_samples = np.array(config.intent(U_samples[0], U_samples[1]))
    U_samples_list[i] = U_samples
    I_samples_list[i] = I_samples

print(U_samples_list.shape)
print(I_samples_list[0].shape)
print(I_samples.shape)

#%%
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

# Observation & CF data
observed_successes = (s + np.diag(data_obs[:, 1])).astype(int)  # Successes
observed_failures = (f + np.diag(data_obs[:, 0])).astype(int)  # Failures
N_data = (observed_successes + observed_failures).astype(int)  # Total

# Experimental data
experimental_successes = data_exp[:, 1].astype(int)
experimental_failures = data_exp[:, 0].astype(int)
N_data_exp = np.sum(data_exp, axis=1).astype(int)  # Total experimental data for each action

# Observed data success rate (4 x 4)
p_wins = observed_successes / N_data

#%%
# -------------------------------------------- #
# OBTAIN DUMMY EXPLORATION DATA                #
# -------------------------------------------- #
# Instantiate (sample) confounders for all T steps
U_samples = np.random.randint(2, size=(config.N_CONFOUNDERS, config.T))
I_samples = np.array(config.intent(U_samples[0], U_samples[1]))

for t in range(200):
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
    observed_successes[action, I] += reward
    observed_failures[action, I] += 1 - reward
    N_data = observed_successes + observed_failures
    p_wins = observed_successes / N_data
    # p_wins is y_observed_data (observational & exploration)


#%%
# -------------------------------------------- #
# MCMC                                         #
# -------------------------------------------- #

# Model with experimental data - wrong
model = pm.Model()
with model:

    # Priors for unknown model parameters

    # c parameters
    hyper_mu = pm.Normal('hyper_mu', mu=0, sd=10)
    hyper_sd = pm.Gamma('hyper_sd', alpha=0.01, beta=0.001)
    c = pm.Normal('c', mu=hyper_mu, sd=hyper_sd, shape=(config.K, config.K))

    p = config.sigmoid(c)
    # Likelihood (sampling distribution) of observations
    #gamma = pm.Binomial('gamma', n=N_data, p=p, observed=observed_successes)
    gamma = pm.Potential('gamma', observed_successes * tt.log(p) + observed_failures * tt.log(1 - p))

    # Nuisance parameter
    theta = pm.Dirichlet('theta', data_exp[:, 0], shape=(1, 4))  # same shape as "beta" before
    # theta = np.reshape(E_exp, (1, 4))

    def joint(gamma, theta):
        return gamma * theta

    L = pm.DensityDist('L', joint, observed={'gamma': gamma, 'theta': theta})

#%%
print(tt.sum([1, 1, 1, 1]))
print(data_exp[:, 0])
print(np.ones(4))

#%%
# Fixed model
model = pm.Model()
with model:

    # Priors for unknown model parameters
    hyper_mu = pm.Normal('hyper_mu', mu=0, sd=10)
    hyper_sd = pm.Gamma('hyper_sd', alpha=0.01, beta=0.001)
    c = pm.Normal('c', mu=hyper_mu, sd=hyper_sd, shape=(config.K, config.K))

    # Prior for nuisance parameter
    theta0 = np.ones(4)
    theta = pm.Dirichlet('theta', theta0, shape=(1, 4))

    p = config.sigmoid(c)
    # Likelihood (sampling distribution) of observations
    # Observational and counterfactual
    L_obs_and_cf = pm.Binomial('L_obs_and_cf', n=N_data, p=p, observed=observed_successes)
    #L_obs_and_cf = pm.Potential('L_obs_and_cf', observed_successes * tt.log(p) + observed_failures * tt.log(1 - p))

    # Experimental
    p_exp = pm.math.sum(p * theta, axis=1)
    L_int = pm.Binomial('L_int', n=N_data_exp, p=p_exp, observed=experimental_successes)
    #L_int = pm.Potential('L_int', experimental_successes * tt.log(p_exp) + experimental_failures * tt.log(1 - p_exp))

    # theta = pm.Dirichlet('theta', data_exp[:, 0], shape=(1, 4))  # same shape as "beta" before
    # theta = np.reshape(E_exp, (1, 4))

    # def joint(gamma, theta):
    #     return gamma * theta
    #
    # L = pm.DensityDist('L', joint, observed={'gamma': gamma, 'theta': theta})
#%%
model.check_test_point()

#%%
p_exp.tag.test_value

#%%
# graph = pm.graph.graph(model)
pydotprint(model.logpt)
plt.show()

#%%

# OLD MODEL
# model = pm.Model()
# with model:
#
#     # Priors for unknown model parameters
#     hyper_mu = pm.Normal('hyper_mu', mu=0, sd=10)
#     hyper_sd = pm.Gamma('hyper_sd', alpha=0.01, beta=0.001)
#     c = pm.Normal('c', mu=hyper_mu, sd=hyper_sd, shape=(config.K, config.K))
#
#     p = config.sigmoid(c)
#
#     # Likelihood (sampling distribution) of observations
#     L = pm.Binomial('L', n=N_data, p=p, observed=observed_successes)

#%%

# OLD OLD MODEL
model = pm.Model()
with model:

    hyper_mu = pm.Normal('hyper_mu', mu=0, sd=10)
    hyper_sd = pm.Gamma('hyper_sd', alpha=config.ALPHA_HYPER_GAMMA_SD,
                        beta=config.BETA_HYPER_GAMMA_SD)
    a = pm.Normal('a', mu=hyper_mu, sd=hyper_sd, shape=(config.K, 1))
    b = pm.Normal('b', mu=hyper_mu, sd=hyper_sd, shape=(1, config.K))
    offset = pm.Normal('offset', mu=hyper_mu, sd=hyper_sd)

    p = config.sigmoid(a + b + offset)

    gamma = pm.Potential('gamma', observed_successes * tt.log(p) + observed_failures * tt.log(1 - p))

    # Nuisance parameter
    theta = pm.Dirichlet('theta', data_exp[:, 0], shape=(1, 4))  # same shape as "beta" before
    # theta = np.reshape(E_exp, (1, 4))

    def joint(gamma, theta):
        return gamma * theta

    L = pm.DensityDist('L', joint, observed={'gamma': gamma, 'theta': theta})

#%%
# MAP
MAP = pm.find_MAP(model=model)  # Find starting point of MCMC
#a_ = np.tile(np.squeeze(MAP['a']), (4, 1)).T
#b_ = np.tile(np.squeeze(MAP['b']), (4, 1))
c_ = np.squeeze(MAP['c'])
#intercept_ = MAP['intercept']
# y_MAP = config.sigmoid(a_ + b_ + c_ + intercept_)
y_MAP = config.sigmoid(c_)
print(y_MAP)

#%%
# Draw posterior samples
#tic()
with model:
    # THIS TAKES A WHILE TO RUN!
    trace = pm.sample(1200, nuts_kwargs=dict(target_accept=.9,
                      max_treedepth=20), chains=config.N_MCMC_CHAINS, init='adapt_diag')
#toc()

#%%
pm.traceplot(trace)
plt.show()

#%%
# Posterior from PPC sampled observations
ppc = pm.sample_ppc(trace, samples=500, model=model)
ppc_result = np.mean(ppc['L'], axis=0) / N_data

#%%
# Posteiror from trace
c_post = np.mean(trace[:100]['c'], axis=0)
y_post = config.sigmoid(c_post)
print(y_post.round(2))

# FOR OLD OLD MODEL
# a_post = np.array(np.mean(trace[:200]['a'], axis=0))
# b_post = np.array(np.mean(trace[:200]['b'], axis=0))
# intercept_post = np.mean(trace[:200]['offset'], axis=0)
# y_post = config.sigmoid(a_post + b_post + intercept_post)
# print(y_post.round(2))

#%%
# Tests
print(trace[-1]['c'])
c_sample = trace[-2]['c']
c_sample = np.mean(trace[:-1]['c'], axis=0)
y_post_sample = config.sigmoid(c_sample)
print(y_post_sample.round(2))

theta_post = np.mean(trace[:100]['theta'], axis=0)
print(theta_post)

#%%
# -------------------------------------------- #
# COMPARE WITH ACTUAL DATA                     #
# -------------------------------------------- #
round = 2
print('y_truth = \n {} \n'.format(config.THETA.round(round)))
print('y_observed = \n {} \n'.format(p_wins.round(round)))
# print('y_logistic_unregularised = \n {} \n'.format(y_out_logistic.round(round)))
# print('y_logistic_sgd_regularised = \n {} \n'.format(y_out_sgd.round(round)))
# print('y_posterior_ppc = \n {} \n'.format(ppc_result.round(round)))
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
#%%

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

import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt

from config import Config
config = Config()


def sigmoid(x):
    """
    Numerically stable sigmoid function
    """
    z = np.exp(x)
    return z / (1 + z)


# -------------------------------------------- #
# Load experimental and observational data     #
# -------------------------------------------- #

#data_exp = np.load('data_exp.npy')
data_obs = np.load('data_obs.npy')

E_obs = data_obs[:, 1] / (data_obs[:, 0] + data_obs[:, 1])     # E_obs(Y|X)
#E_exp = data_exp[:, 1] / (data_exp[:, 0] + data_exp[:, 1])     # E_exp(Y|do(X))
# P(X) from observational data, X=intent
p_int = np.sum(data_obs, axis=1) / np.sum(data_obs)

s = np.ones((config.K, config.K))  # successes
f = np.ones((config.K, config.K))  # failures

# Exploration + observation data matrix
s_with_obs = (s + np.diag(data_obs[:, 1])).astype(int)
f_with_obs = (f + np.diag(data_obs[:, 0])).astype(int)

N_data = (s_with_obs + f_with_obs).astype(int)

p_wins = s_with_obs / N_data

# -------------------------------------------- #
# SAMPLE SOME FIRST                            #
# -------------------------------------------- #
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

    N_data = s_with_obs + f_with_obs


# # ------------------------------------------------- #
# # CREATE BERNOULLI INPUT FROM SUCCESS PROPORTION    #
# # ------------------------------------------------- #
# N_bernoulli_sims = 500
# data = np.ones((config.K, config.K, N_bernoulli_sims))
# change_to_zeros = np.round((1 - p_wins) * N_bernoulli_sims).astype(int)
# data[change_to_zeros[:, :, None] > np.arange(data.shape[-1])] = 0
# data = np.transpose(data, (2, 0, 1))  # Swap axes
# data = np.take(data, np.random.rand(data.shape[0]).argsort(), axis=0)

# ------------------------------------------------- #
# USE PYMC3 TO CREATE POSTERIOR AND SAMPLE FROM IT  #
# ------------------------------------------------- #

model = pm.Model()
with model:

    # Priors for unknown model parameters
    a = pm.Normal('a', mu=0, sd=10, shape=(config.K, 1))
    b = pm.Normal('b', mu=0, sd=10, shape=(1, config.K))
    offset = pm.Normal('offset', mu=0, sd=10)

    # p = pm.Deterministic('p', sigmoid(a + b + offset))
    p = sigmoid(a + b + offset)

    # Likelihood (sampling distribution) of observations
    pm.Binomial('L', n=N_data, p=p, observed=s_with_obs)

with model:
    # draw posterior samples
    trace = pm.sample(2000, nuts_kwargs=dict(target_accept=.9,
                                             max_treedepth=20), chains=3)

# ------------------------------------------------- #
# RESULTS OF TRACE                                  #
# ------------------------------------------------- #

pm.traceplot(trace)
plt.show()
pm.summary(trace).round(5)

a_post = np.array(np.mean(trace[:200]['a'], axis=0))
b_post = np.array(np.mean(trace[:200]['b'], axis=0))
offset_post = np.mean(trace[:200]['offset'], axis=0)
print(a_post)
print(b_post)
print(offset_post)

# INFO: https://docs.pymc.io/notebooks/posterior_predictive.html
ppc = pm.sample_ppc(trace, samples=500, model=model)
# ppc_result = np.mean(ppc['L'], axis=(0, 1)).round(2)  # BERNOULLI
ppc_result = (np.mean(ppc['L'], axis=0) / N_data).round(3)  # BINOMIAL
print('ppc_result = \n', ppc_result)

y_post = sigmoid(a_post + b_post + offset_post)
print('y_post = \n', y_post.round(3))

print('p_wins = \n', p_wins.round(3))
print('THETA = \n', config.THETA)

# TEST FOR SAMPLING ONCE FROM POSTERIOR
samp = pm.sample_ppc(trace, samples=1, model=model)
print(samp['L'].shape)
print(np.mean(samp['L'], axis=(0, 1)))


# ------------------------------
# import scipy
#
# # set up constants
# p_true = 0.1
# N = 3000
# observed = scipy.stats.bernoulli.rvs(p_true, size=N)
# print(observed.shape)
#
# print(p_wins)

# model = pm.Model()
# with model:
#
#     # Priors for unknown model parameters
#     a = pm.Normal('a', mu=0, sd=10, shape=(config.K, 1))
#     b = pm.Normal('b', mu=0, sd=10, shape=(1, config.K))
#     offset = pm.Normal('offset', mu=0, sd=10)
#
#     p = pm.Deterministic('p', sigmoid(a + b + offset))
#
#     L = pm.Binomial('L', N_data, p, observed=s_with_obs)
#
# with model:
#     # draw posterior samples
#     trace = pm.sample(2000, nuts_kwargs=dict(target_accept=.9,
#                                              max_treedepth=20), chains=3)
#
# pm.traceplot(trace)
# plt.show()
# pm.summary(trace).round(5)
#
# a_post = np.array(np.mean(trace[:100]['a'], axis=0))
# b_post = np.array(np.mean(trace[:100]['b'], axis=0))
# offset_post = np.mean(trace[:100]['offset'], axis=0)
# # print(a_post)
# # print(b_post)
# # print(offset_post)
#
# ppc = pm.sample_ppc(trace, samples=500, model=model)
# #print(ppc['observations'].shape)
# ppc_sum = np.mean(ppc['observations'], axis=0)
# ppc_result = ppc_sum / N_data
# #print(np.asarray(ppc['L']).shape)
# print('ppc_result = \n', ppc_result.round(3))
# y_post = sigmoid(a_post + b_post + offset_post)
# print('y_post = \n', y_post.round(3))
# print('p_wins = \n', p_wins.round(3))
# print('THETA = \n', config.THETA)
#
# sample = pm.sample_ppc(trace, samples=1, model=model)
# print(sample['observations'])
# print(N_data)
# print((np.mean(sample['observations'], axis=0) / N_data))

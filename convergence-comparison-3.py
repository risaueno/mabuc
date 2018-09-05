#%%
import numpy as np
import pymc3 as pm
import theano.tensor as tt
import matplotlib.pyplot as plt
from helper import tic, toc, smooth
from theano.printing import pydotprint
import os
from datasim import create_data, generate_and_save_n_data, sample, create_shuffled_data

from config import Config
config = Config()

plot_traceplot = False
run_gradient_descent = False

import datetime
stamp = str(datetime.datetime.now()).split('.')[0]
stamp

stamp = stamp.replace(':','-')
stamp

stamp = stamp.replace(' ','-')
stamp

#%%
theta_list = np.load('theta_list.npy')
intent = np.array([0,2,1,3,2,0,1,2])
action = np.array([2,0,1,2,2,1,0,3])

th = theta_list[0]
th

optimal_rewards = np.max(th[:, intent], axis=0)



for th in theta_list:
    th[:, intent]


#%%

ihdp_payout = np.load('ihdp_payout.npy')
ihdp_payout.round(2)

ihdp_exp_p = np.load('ihdp_exp_p.npy')
ihdp_exp_p.round(2)

np.load('ihdp_exp_success.npy') + np.load('ihdp_exp_failure.npy')

np.load('ihdp_obs_success.npy') + np.load('ihdp_obs_failure.npy')

#%%

data_exp_list, data_obs_list, theta_list = create_shuffled_data(20, noisy=False)

theta_list[0]

data_obs_list[0]

data_exp_list[0]


np.save('data_exp_list', data_exp_list)
np.save('data_obs_list', data_obs_list)
np.save('theta_list', theta_list)

len(theta_list)

data_exp_list = np.load('data_exp_list.npy')
data_obs_list = np.load('data_obs_list.npy')
theta_list = np.load('theta_list.npy')

data_exp_list = []
data_obs_list = []
theta_list = []

data_exp_list_clean = []
data_obs_list_clean = []

if config.K == 6:
    payouts = np.tile(np.array([0.2, 0.3, 0.3, 0.5, 0.6, 0.4]), (6, 1))
elif config.K == 4:
    payouts = np.tile(np.array([0.2, 0.3, 0.6, 0.5]), (4, 1))
else:
    print("K must be 4 or 6")

payouts

N = 5
for _ in range(N):
    shuffled_payout = config.crazyshuffle(payouts)
    theta_list.append(shuffled_payout)
    exp, obs = create_data(theta=shuffled_payout, noisy=True)
    exp_clean, obs_clean = create_data(theta=shuffled_payout, noisy=False)
    data_exp_list.append(exp)
    data_obs_list.append(obs)
    data_exp_list_clean.append(exp_clean)
    data_obs_list_clean.append(obs_clean)

data_exp_list[0]

data_obs_list[0]

theta_list[0]

theta = theta_list[0]
obs_clean = np.zeros((config.K, 2))
for i in range(config.K):
    obs_clean[i, :] = [(1 - theta[i, i]) * config.SAMPLES_PER_ARM, theta[i, i] * config.SAMPLES_PER_ARM]
obs_clean

exp_clean = np.zeros((config.K, 2))
for i in range(config.K):
    exp_clean[i, :] = [(1 - np.mean(theta[i, :])) * config.SAMPLES_PER_ARM, np.mean(theta[i, :]) * config.SAMPLES_PER_ARM]
exp_clean

#%%

Prob_log = np.zeros((5, 10))
Prob_log[0] = np.random.randint(5, size=(10))
Prob_log[1] = np.random.randint(5, size=(10))
Prob_log

data = Prob_log[:(2)]
data

np.std(data, axis=0, ddof=1)

#%%
actions = np.arange(config.K)
actions
T = 17
np.resize(actions, T)

N = 10
I_samps = np.zeros((N, T))

I_patterns = np.zeros((config.K, config.K)).astype(int)
for i in range(config.K):
    I_patterns[i] = np.roll(np.arange(config.K, dtype=np.int), -i)
I_patterns
np.ceil(T / config.K).astype(int)

I_patterns = np.tile(I_patterns, np.ceil(T / config.K).astype(int))
I_patterns

I_patterns = np.tile(I_patterns, (np.ceil(N / config.K).astype(int), 1))
I_patterns

I_samples = I_patterns[:N, :T]
I_samples

for i in range(10):
    print(np.roll(I_patterns.T, i, axis=1)[0])

for n in N:
    for i in range(config.K):
        next = np.roll(np.arange(config.K, dtype=np.int), -i)
        I = np.resize(next, T)

#%%
data_exp, data_obs = create_data()
# np.save('data_exp_6', data_exp)
# np.save('data_obs_6', data_obs)

#%%

payouts = np.tile(np.array([0.2, 0.3, 0.3, 0.5, 0.6]), (5, 1)).T
payouts
np.random.shuffle(payouts.T)
payouts


def crazyshuffle(arr):
    x, y = arr.shape
    rows = np.indices((x,y))[0]
    cols = [np.random.permutation(y) for _ in range(x)]
    return arr[rows, cols].T

payouts = np.tile(np.array([0.2, 0.3, 0.3, 0.5, 0.6, 0.4]), (6, 1))
crazyshuffle(payouts)

#%%
payouts = config.THETA
print(config.THETA)

Intents = np.array([0,2,3,1,2,1])
Actions = np.array([3,2,3,1,0,0])

print(payouts[:, Intents])

optimal_rewards = np.max(payouts[:, Intents], axis=0)
action_rewards = payouts[Actions, Intents]

regret = optimal_rewards - action_rewards
cum_regret = np.cumsum(regret)

print(optimal_rewards)
print(action_rewards)

print(regret)
print(cum_regret)

#%%
payouts = config.THETA
print(config.THETA)
np.random.shuffle(np.transpose(payouts))
print(payouts)

intents = np.arange(4)
actions = np.arange(4)
rewards = np.arange(2)
IAY_permutations = np.array(np.meshgrid(intents, actions, rewards)).T.reshape(-1, 3)

print(intents)
print(rewards)

I = 2
A = 3
Y = 1

intent_2 = config.sigmoid(I + A + Y)
print(intent_2)

# Generate all permulations
permutations_all = np.array(np.meshgrid(intents, actions, rewards)).T.reshape(-1, 3)
print(permutations_all)

# item_index = np.where(permutations_all == [A, I, Y])
item_index = np.squeeze(np.where((permutations_all == [I, A, Y]).all(axis=1)))
print(item_index)

# SOFTMAX
# combi_sum = np.sum(permutations_all, axis=1)
# print(combi_sum)
# softmax_sum = np.sum(np.exp(combi_sum))
# softmax = np.exp(combi_sum) / softmax_sum
# print(softmax)

second_intent_probs = np.random.dirichlet(np.ones(4), size=32)
# print(random_probs[0, :])
second_intent = np.random.choice(4, 1, p=second_intent_probs[item_index, :])
print(second_intent_probs[item_index, :])
print(second_intent)
second_intent_list = np.squeeze([np.random.choice(4, 1, p=second_intent_probs[i, :]).astype(int) for i in np.arange(32)])
print(second_intent_list)
# print(np.random.choice(4, 32, p=[0.1, 0, 0.3, 0.6, 0]))

# SIMULATE DATA
# INTENT = ACTION, Y = 1: good psychology,
# INTENT = ACTION, Y = 0: below average psychology,
# INTENT != ACTION, Y = 1: average psychology,
# INTENT != ACTION, Y = 0: bad psychology

np.random.seed(100)
next_intent_probs = np.random.dirichlet(np.ones(4), size=permutations_all.shape[0])
print(next_intent_probs.shape)

#%%

IAYs = []

for i in range(30):
    IAYs.append(i)

IAYs = np.array(IAYs)
print(IAYs)

IAYs.reshape((-1, 3))
print(IAYs)

#%%
I_samples = np.load('I_samples_2018-08-03 08:35:17.npy')
print(I_samples)
print(I_samples.shape)
unique, counts = np.unique(I_samples, return_counts=True)
print(dict(zip(unique, counts)))
test = np.tile(I_samples, (2, 1))
print(test.shape)

#%%
theta1_ = np.array([[0.2, 0.3, 0.5, 0.6],
                    [0.6, 0.2, 0.3, 0.5],
                    [0.5, 0.6, 0.2, 0.3],
                    [0.3, 0.5, 0.6, 0.2]])

theta2_ = np.array([[0.2, 0.3, 0.5, 0.6],
                    [0.6, 0.2, 0.3, 0.5],
                    [0.5, 0.6, 0.2, 0.3],
                    [0.3, 0.5, 0.6, 0.2]])

print(theta1_.shape)
THETA = np.stack((theta1_, theta2_))
print(THETA.shape)

#%%
T = 20
Intent = np.random.randint(4, size=T)       # Intent (0-3) seen for each timestep
Action = np.random.randint(4, size=T)       # Action (0-3) taken for each timestep
Reward = np.random.randint(2, size=T)       # Reward (0,1) observed for each timestep

intents = np.arange(4)
actions = np.arange(4)
rewards = np.arange(2)
IAY_permutations = np.array(np.meshgrid(intents, actions, rewards)).T.reshape(-1, 3)

IAYs = np.stack((Intent, Action, Reward)).T  # (T, 3)

# Find indices in permutation list for each (I, A, Y)
permutation_indices = np.where((IAY_permutations == IAYs[:, None]).all(-1))[1]
print(permutation_indices.shape)

# Get class probabilities list for each permutation
next_intent_probs = np.random.dirichlet(np.ones(4), size=IAY_permutations.shape[0])
class_probabilities = next_intent_probs
print(class_probabilities.shape)

# Get list of next intents
next_intent_list = np.squeeze([np.random.choice(config.K, 1, p=class_probabilities[i, :]).astype(int)
                               for i in permutation_indices])
print(next_intent_list)

intents = np.arange(4)
actions = np.arange(4)
rewards = np.arange(2)
permutations_all = np.array(np.meshgrid(intents, actions, rewards)).T.reshape(-1, 3)
permutations_all

wheres = np.where((permutations_all == IAYs[:, None]).all(-1))[1]
wheres

wheres = np.where(permutations_all == IAYs.T)
wheres

#%%

# Create I samples
filename = 'I_samples_TEST2'
dir = 'data_folder'

N = 5
T = 1500

U_samples_list = np.empty((N, config.N_CONFOUNDERS, T))
I_samples_list = np.empty((N, T))

for i in range(N):
    U_samples = np.random.randint(2, size=(config.N_CONFOUNDERS, T))
    I_samples = np.array(config.intent(U_samples[0], U_samples[1]))
    U_samples_list[i] = U_samples
    I_samples_list[i] = I_samples

#print(U_samples_list.shape)
#print(I_samples_list[0].shape)
print(I_samples_list.shape)
print(I_samples_list)
unique, counts = np.unique(I_samples_list[:, :50], return_counts=True)
print(dict(zip(unique, counts)))


np.save(os.path.join(dir, filename), I_samples_list)
#print(I_samples_list.shape)

#%%
# -------------------------------------------- #
# Load experimental and observational data     #
# -------------------------------------------- #

data_obs = np.load('data_obs.npy')  # Observational data
data_exp = np.load('data_exp.npy')  # Experimental data (not used here)

p_int = np.sum(data_obs, axis=1) / np.sum(data_obs)
print(p_int)

p_int_theta = p_int[:, np.newaxis].T
print(p_int_theta)

p_int_theta = np.tile(p_int, (config.K, 1))
print(p_int_theta)

s = np.ones((config.K, config.K))  # Initialise exploration successes

s * p_int_theta

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

model.check_test_point()

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

#%%

save_dir = 'data_folder'
PProb_file = 'PProb_RDC_original++_2018-08-07 11:27:42.npy'
Regret_file = 'Regret_MCMC+, a=100, b=10_2018-08-06 18:03:19.npy'
PProb_file2 = 'PProb_MCMC++, a=100, b=10_2018-08-05 13:10:57.npy'
Regret_file2 = 'Regret_MCMC+, a=100, b=10_2018-08-06 18:03:19.npy'
PProb_file3 = 'PProb_MCMC+, a=100, b=10_2018-08-06 18:03:19.npy'
Regret_file3 = 'Regret_MCMC+, a=100, b=10_2018-08-06 18:03:19.npy'

PProb = np.load(os.path.join(save_dir, PProb_file))
Regret = np.load(os.path.join(save_dir, Regret_file))
PProb2 = np.load(os.path.join(save_dir, PProb_file2))
Regret2 = np.load(os.path.join(save_dir, Regret_file2))
PProb3 = np.load(os.path.join(save_dir, PProb_file3))
Regret3 = np.load(os.path.join(save_dir, Regret_file3))

plt.figure()

max_x_range = 200

ax = plt.subplot(211)
ax.plot(smooth(PProb), label='1')
ax.plot(smooth(PProb2), label='2')
ax.plot(smooth(PProb3), label='3')
ax.set_yticks(np.arange(0, 1., 0.1))
ax.set_xticks(np.arange(0, max_x_range, 10))
ax.grid()

ax2 = plt.subplot(212)
ax2.plot(smooth(Regret), label=Regret_file)
ax2.set_yticks(np.arange(0, 100, 10))
ax2.set_xticks(np.arange(0, max_x_range, 10))
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

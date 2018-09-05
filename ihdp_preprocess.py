#%%
import numpy as np
import pymc3 as pm
import theano.tensor as tt
import matplotlib.pyplot as plt
from helper import tic, toc, smooth
from theano.printing import pydotprint
import os
from datasim import create_data
import pandas as pd

from config import Config
config = Config()

plot_traceplot = False
run_gradient_descent = False

import datetime

print(str(datetime.datetime.now()).split('.')[0])

#%%
# Load data (Y, X, RACE, EDUCATION)
# which(ns == "iqsb.36"), which(ns == "ncdctt"), which(ns == "momrace"), which(ns == "momed")
# 0: Y (IQ)
# 1: X (TREATMENT DAYS)

# Suppress scientific notation when printing
np.set_printoptions(suppress=True)

df = pd.read_csv('ihdp_all.dat', sep=",", header=None)
ihdp_data = df.as_matrix().round(5)
# np.save('ihdp', ihdp_data)
ihdp_data.shape
# ihdp_data = np.fromfile('ihdp.dat', dtype=float)

# BINARIZE Y
i_y = 0
# y_cutoff = 100
y_cutoff = np.percentile(ihdp_data[:, i_y], 50)  # IQ = 93

ihdp_data[np.where(ihdp_data[:, 0] <= y_cutoff), 0] = 0
ihdp_data[np.where(ihdp_data[:, 0] > y_cutoff), 0] = 1
# ihdp_data[:10, :3]

#%%

# DISCRETIZE X (ACTION)
i_x = 1  # INDEX
n_action = 4

# # SPLIT BY 4
# x_cutoff = np.max(ihdp_data[:, i_x]) / n_action
# x_cutoff
# indices = []
# indices.append(np.where(ihdp_data[:, i_x] <= x_cutoff))
# for i in range(n_action - 2):
#     indices.append(np.where((ihdp_data[:, i_x] > x_cutoff * (i + 1)) & (ihdp_data[:, i_x] <= x_cutoff * (i + 2))))
# indices.append(np.where(ihdp_data[:, i_x] > (x_cutoff * (n_action - 1))))
#
# ihdp_data[indices[0], i_x] = 0
# for i in range(n_action - 1):
#     ihdp_data[indices[i], i_x] = i
# ihdp_data[indices[-1], i_x] = (n_action - 1)

# OR SPLIT BY QUARTILE

np.percentile(ihdp_data[:, i_x], 25)
np.percentile(ihdp_data[:, i_x], 50)
np.percentile(ihdp_data[:, i_x], 75)

(390 - 230) / 2

np.max(ihdp_data[:, i_x])
np.min(ihdp_data[:, i_x])

# Delete data where patients didn't choose to treat themselves?
# np.array(np.where(ihdp_data[:, i_x] == 0)).shape
# ihdp_data.shape
# ihdp_data = np.delete(ihdp_data, np.where(ihdp_data[:, i_x] == 0), axis=0)
# ihdp_data.shape

# Get indices first
indices = []
indices.append(np.where(ihdp_data[:, i_x] <= np.percentile(ihdp_data[:, i_x], 25)))
indices.append(np.where((ihdp_data[:, i_x] > np.percentile(ihdp_data[:, i_x], 25)) &
                        (ihdp_data[:, i_x] <= np.percentile(ihdp_data[:, i_x], 50))))
indices.append(np.where((ihdp_data[:, i_x] > np.percentile(ihdp_data[:, i_x], 50)) &
                        (ihdp_data[:, i_x] <= np.percentile(ihdp_data[:, i_x], 75))))
indices.append(np.where(ihdp_data[:, i_x] > np.percentile(ihdp_data[:, i_x], 75)))

# indices = []
# indices.append(np.where(ihdp_data[:, i_x] <= 2.3))
# indices.append(np.where((ihdp_data[:, i_x] > 2.3) &
#                         (ihdp_data[:, i_x] <= 3.1)))
# indices.append(np.where((ihdp_data[:, i_x] > 3.1) &
#                         (ihdp_data[:, i_x] <= 3.9)))
# indices.append(np.where(ihdp_data[:, i_x] > 3.9))


# Select from indices
ihdp_data[indices[0], i_x] = 0
for i in range(n_action - 1):
    ihdp_data[indices[i], i_x] = i
ihdp_data[indices[-1], i_x] = (n_action - 1)

# ----------

# Split data by action taken
ihdp_0 = ihdp_data[np.where(ihdp_data[:, i_x] == 0)]
ihdp_1 = ihdp_data[np.where(ihdp_data[:, i_x] == 1)]
ihdp_2 = ihdp_data[np.where(ihdp_data[:, i_x] == 2)]
ihdp_3 = ihdp_data[np.where(ihdp_data[:, i_x] == 3)]

ihdp_separated = [ihdp_0, ihdp_1, ihdp_2, ihdp_3]

# Average reward observed for each action bucket
action_equal_intent = np.array([np.mean(ihdp_0[:, i_y]), np.mean(ihdp_1[:, i_y]), np.mean(ihdp_2[:, i_y]), np.mean(ihdp_3[:, i_y])])
action_equal_intent

ihdp_0.shape
ihdp_1.shape
ihdp_2.shape
ihdp_3.shape

# Get counts for observational successes and failures
successes = np.zeros(n_action)
failures = np.zeros(n_action)
for i in range(n_action):
    successes[i] = np.sum(ihdp_separated[i][:, i_y])
    failures[i] = ihdp_separated[i].shape[0] - successes[i]
successes
failures
successes / (failures + successes)

# Collect data for just covariates (without x, y)
cov_0 = ihdp_0[:, 2:]
cov_1 = ihdp_1[:, 2:]
cov_2 = ihdp_2[:, 2:]
cov_3 = ihdp_3[:, 2:]

cov_separated = [cov_0, cov_1, cov_2, cov_3]

cov_0.shape
cov_1.shape
cov_2.shape
cov_3.shape

#%% TEST
# # For cov_0[0]
# # point = cov_0[0]
# # squared_distance = np.array([np.sum((point - cov_1[i]) ** 2) for i in range(cov_1.shape[0])])
# # np.argmax(squared_distance)
#
# # # For cov_0
# # action_batch = cov_separated[0]
# #
# # compare = [1, 2, 3]
# #
# # cov = cov_separated[1]
# # cov.shape
# #
# # closest_points = np.zeros(action_batch.shape[0]).astype(int)
# # closest_points.shape
# #
# # for point in range(action_batch.shape[0]):
# #     squared_distance = np.array([np.sum((action_batch[point] - cov[i]) ** 2) for i in range(cov.shape[0])])
# #     closest_point_index = np.argmin(squared_distance)
# #     closest_points[point] = closest_point_index
# #
# # closest_points.shape
#
# # idx = 1
# # actions = np.arange(n_action)
# # np.delete(actions, idx)
#
# actions = np.arange(n_action)  # 0123
# idx = 1
# action_batch = cov_separated[idx]  # Covariates for A=0
# compare = np.delete(actions, idx)  # 123
# compare
#
# c = 1
# cov = cov_separated[c]   # 40 points in A=1
# cov.shape
#
# point = 0
# cov_point = action_batch[point]   # Choose datapoint within covariates for A=0
# cov_point.shape
# squared_distance = np.array([np.sum((action_batch[point] - cov[i]) ** 2) for i in range(cov.shape[0])])
# squared_distance.shape
# squared_distance.round(2)
# np.argmin(squared_distance)
#
#
# closest_points = np.zeros(action_batch.shape[0]).astype(int)
#
# # FOR EACH POINT IN A=0 FIND CLOSEST
# for point in range(action_batch.shape[0]):
#     squared_distance = np.array([np.sum((action_batch[point] - cov[i]) ** 2) for i in range(cov.shape[0])])
#     closest_point_index = np.argmin(squared_distance)
#     closest_points[point] = closest_point_index
#
# closest_points
#
# # -----
# actions = np.arange(n_action)  # 0123
# idx = 1
# action_batch = cov_separated[idx]  # Covariates for A=1
# action_batch.shape
#
# np.mean(ihdp_separated[idx][:, i_y])
#
# compare = np.delete(actions, idx)  # 123
# compare
#
# c = 1
# cov = cov_separated[compare[c]]   # compare with A = 2
# cov.shape
#
# closest_points = np.zeros(action_batch.shape[0]).astype(int)
# reward_of_closest_points = np.zeros(action_batch.shape[0]).astype(int)
#
# # For point 0 in A=0
# squared_distance = np.array([np.sum((action_batch[1] - cov[i]) ** 2) for i in range(cov.shape[0])])
# squared_distance
#
# # FOR EACH POINT IN A=0 FIND CLOSEST
# for point in range(action_batch.shape[0]):
#     squared_distance = np.array([np.sum((action_batch[point] - cov[i]) ** 2) for i in range(cov.shape[0])])
#     closest_point_index = np.argmin(squared_distance)
#     closest_points[point] = closest_point_index
#
#     reward_of_closest_points[point] = ihdp_separated[compare[c]][closest_point_index, i_y]
#
# closest_points
# reward_of_closest_points
# np.mean(reward_of_closest_points)
# # ---------------------

#%% CREATE OBSERvATIONAL DATA

best_matching_indices = []
best_matching_rewards = []
actions = np.arange(n_action)

for idx in range(n_action):

    action_batch = cov_separated[idx]
    compare = np.delete(actions, idx)

    closest = []
    rewards_closest = []

    for c in compare:

        cov = cov_separated[c]

        closest_points = np.zeros(action_batch.shape[0]).astype(int)
        reward_of_closest_points = np.zeros(action_batch.shape[0]).astype(int)

        for point in range(action_batch.shape[0]):
            squared_distance = np.array([np.sum((action_batch[point] - cov[i]) ** 2) for i in range(cov.shape[0])])
            closest_point_index = np.argmin(squared_distance)
            closest_points[point] = closest_point_index

            reward_of_closest_points[point] = ihdp_separated[c][closest_point_index, i_y]

        closest.append(closest_points)
        rewards_closest.append(reward_of_closest_points)

    best_matching_indices.append(closest)
    best_matching_rewards.append(rewards_closest)

# Put together mean of counterfactual (expected) rewards
expected_rewards = np.zeros((n_action, n_action - 1))
for idx in range(n_action):
    column = best_matching_rewards[idx]
    average_rewards = np.zeros(n_action - 1)
    for i in range(n_action - 1):
        average_rewards[i] = np.mean(column[i])
    expected_rewards[idx] = average_rewards

# COMBINE ALL TOGETHER

obs_T = np.diag(action_equal_intent)
for i in range(n_action):
    compare = np.delete(actions, i)  # 123
    for c in range(n_action - 1):
        a_cf = compare[c]
        obs_T[i, a_cf] = expected_rewards[i, c]

obs = obs_T.T
obs.round(3)

# PErcentiles ***
# array([[ 0.356,  0.264,  0.209,  0.425],
#        [ 0.598,  0.46 ,  0.5  ,  0.471],
#        [ 0.552,  0.414,  0.488,  0.414],
#        [ 0.667,  0.586,  0.547,  0.667]])

# Equal buckets
# array([[ 0.36 ,  0.314,  0.222,  0.469],
#        [ 0.629,  0.471,  0.452,  0.469],
#        [ 0.551,  0.471,  0.484,  0.383],
#        [ 0.674,  0.549,  0.54 ,  0.667]])

# Equal buckets, 0 treatment excluded
# array([[ 0.295,  0.314,  0.278,  0.383],
#        [ 0.607*,  0.471,  0.452,  0.469],
#        [ 0.525,  0.471,  0.484,  0.383],
#        [ 0.607*,  0.549,  0.54 ,  0.667]])

successes
failures

exp = np.mean(obs, axis=1)
exp

#%%
exp_successes = np.array([
obs_T[0] * ihdp_0.shape[0],
obs_T[1] * ihdp_1.shape[0],
obs_T[2] * ihdp_2.shape[0],
obs_T[3] * ihdp_3.shape[0]
]).T

exp_failures = np.array([
(1 - obs_T[0]) * ihdp_0.shape[0],
(1 - obs_T[1]) * ihdp_1.shape[0],
(1 - obs_T[2]) * ihdp_2.shape[0],
(1 - obs_T[3]) * ihdp_3.shape[0]
]).T

exp_successes = np.sum(exp_successes, axis=1).astype(int)
exp_failures = np.sum(exp_failures, axis=1).astype(int)
exp_successes
exp_failures

# exp_successes / (exp_successes + exp_failures)
# exp

#%%  SAVE DATA

np.save('ihdp_payout', obs)
np.save('ihdp_obs_success', successes)
np.save('ihdp_obs_failure', failures)
np.save('ihdp_exp_p', exp)
np.save('ihdp_exp_success', exp_successes)
np.save('ihdp_exp_failures', exp_failures)

#%%

# FIVE ACTIONS VERSION

#%%

# DISCRETIZE X (ACTION)
i_x = 1  # INDEX
n_action = 5

# OR SPLIT BY QUARTILE

np.percentile(ihdp_data[:, i_x], 20)
np.percentile(ihdp_data[:, i_x], 40)
np.percentile(ihdp_data[:, i_x], 60)
np.percentile(ihdp_data[:, i_x], 80)

(400 - 160) / 3

# Get indices first
indices = []
indices.append(np.where(ihdp_data[:, i_x] <= np.percentile(ihdp_data[:, i_x], 20)))
indices.append(np.where((ihdp_data[:, i_x] > np.percentile(ihdp_data[:, i_x], 20)) &
                        (ihdp_data[:, i_x] <= np.percentile(ihdp_data[:, i_x], 40))))
indices.append(np.where((ihdp_data[:, i_x] > np.percentile(ihdp_data[:, i_x], 40)) &
                        (ihdp_data[:, i_x] <= np.percentile(ihdp_data[:, i_x], 60))))
indices.append(np.where((ihdp_data[:, i_x] > np.percentile(ihdp_data[:, i_x], 60)) &
                        (ihdp_data[:, i_x] <= np.percentile(ihdp_data[:, i_x], 80))))
indices.append(np.where(ihdp_data[:, i_x] > np.percentile(ihdp_data[:, i_x], 80)))

# OR

indices = []
indices.append(np.where(ihdp_data[:, i_x] <= 1.6))
indices.append(np.where((ihdp_data[:, i_x] > 1.6) &
                        (ihdp_data[:, i_x] <= 2.4)))
indices.append(np.where((ihdp_data[:, i_x] > 2.4) &
                        (ihdp_data[:, i_x] <= 3.2)))
indices.append(np.where((ihdp_data[:, i_x] > 3.2) &
                        (ihdp_data[:, i_x] <= 4)))
indices.append(np.where(ihdp_data[:, i_x] > 4))

# Select from indices
ihdp_data[indices[0], i_x] = 0
for i in range(n_action - 1):
    ihdp_data[indices[i], i_x] = i
ihdp_data[indices[-1], i_x] = (n_action - 1)

# ----------

# ihdp_data[:20, :3]

ihdp_separated = []
for n in range(n_action):
    ihdp_separated.append(ihdp_data[np.where(ihdp_data[:, i_x] == n)])

# # Split data by action taken
# ihdp_0 = ihdp_data[np.where(ihdp_data[:, i_x] == 0)]
# ihdp_1 = ihdp_data[np.where(ihdp_data[:, i_x] == 1)]
# ihdp_2 = ihdp_data[np.where(ihdp_data[:, i_x] == 2)]
# ihdp_3 = ihdp_data[np.where(ihdp_data[:, i_x] == 3)]
# ihdp_4 = ihdp_data[np.where(ihdp_data[:, i_x] == 4)]
#
# ihdp_separated = [ihdp_0, ihdp_1, ihdp_2, ihdp_3, ihdp_4]

# Average reward observed for each action bucket
action_equal_intent = np.array([np.mean(ihdp_separated[0][:, i_y]),
                                np.mean(ihdp_separated[1][:, i_y]),
                                np.mean(ihdp_separated[2][:, i_y]),
                                np.mean(ihdp_separated[3][:, i_y]),
                                np.mean(ihdp_separated[4][:, i_y])])
action_equal_intent

ihdp_separated[0].shape
ihdp_separated[1].shape
ihdp_separated[2].shape
ihdp_separated[3].shape
ihdp_separated[4].shape

# Get counts for observational successes and failures
successes = np.zeros(n_action)
failures = np.zeros(n_action)
for i in range(n_action):
    successes[i] = np.sum(ihdp_separated[i][:, i_y])
    failures[i] = ihdp_separated[i].shape[0] - successes[i]
successes
failures
successes / (failures + successes)

# Collect data for just covariates (without x, y)

cov_separated = []
for n in range(n_action):
    cov_separated.append(ihdp_separated[n][:, 2:])

# cov_0 = ihdp_0[:, 2:]
# cov_1 = ihdp_1[:, 2:]
# cov_2 = ihdp_2[:, 2:]
# cov_3 = ihdp_3[:, 2:]
# cov_4 = ihdp_4[:, 2:]
#
# cov_separated = [cov_0, cov_1, cov_2, cov_3, cov_4]

cov_separated[0].shape
cov_separated[1].shape
cov_separated[2].shape
cov_separated[3].shape
cov_separated[4].shape

#%% CREATE OBSERvATIONAL DATA

best_matching_indices = []
best_matching_rewards = []
actions = np.arange(n_action)

for idx in range(n_action):
    #print("idx = ", idx)

    action_batch = cov_separated[idx]
    compare = np.delete(actions, idx)

    closest = []
    rewards_closest = []

    for c in compare:
        #print("c = ", c)
        cov = cov_separated[c]

        closest_points = np.zeros(action_batch.shape[0]).astype(int)
        reward_of_closest_points = np.zeros(action_batch.shape[0]).astype(int)

        for point in range(action_batch.shape[0]):
            squared_distance = np.array([np.sum((action_batch[point] - cov[i]) ** 2) for i in range(cov.shape[0])])
            #print("squared_distance length = ", squared_distance.shape)
            closest_point_index = np.argmin(squared_distance)
            closest_points[point] = closest_point_index

            reward_of_closest_points[point] = ihdp_separated[c][closest_point_index, i_y]

        closest.append(closest_points)
        rewards_closest.append(reward_of_closest_points)

    best_matching_indices.append(closest)
    best_matching_rewards.append(rewards_closest)

# Put together mean of counterfactual (expected) rewards
expected_rewards = np.zeros((n_action, n_action - 1))
for idx in range(n_action):
    column = best_matching_rewards[idx]
    average_rewards = np.zeros(n_action - 1)
    for i in range(n_action - 1):
        average_rewards[i] = np.mean(column[i])
    expected_rewards[idx] = average_rewards

# COMBINE ALL TOGETHER

obs_T = np.diag(action_equal_intent)
for i in range(n_action):
    compare = np.delete(actions, i)  # 123
    for c in range(n_action - 1):
        a_cf = compare[c]
        obs_T[i, a_cf] = expected_rewards[i, c]

#%%

obs = obs_T.T
obs.round(3)

# Separate into even treatment days chunks
# array([[ 0.36,  0.18,  0.28,  0.21,  0.42],
#        [ 0.65,  0.36,  0.51,  0.47,  0.61],
#        [ 0.58,  0.46,  0.46,  0.43,  0.48],
#        [ 0.56,  0.39,  0.46,  0.5 ,  0.4 ],
#        [ 0.68,  0.5 ,  0.49,  0.49,  0.69]])

# Serpate into percentiles
# array([[ 0.36,  0.22,  0.23,  0.21,  0.43],
#        [ 0.63,  0.45,  0.37,  0.46,  0.47],
#        [ 0.66,  0.52,  0.49,  0.55,  0.51],
#        [ 0.51,  0.48,  0.37,  0.49,  0.49],
#        [ 0.67,  0.49,  0.49,  0.48,  0.67]])

successes
failures

exp = np.mean(obs, axis=1)
exp

#%%
exp_successes = np.array([
obs_T[0] * ihdp_separated[0].shape[0],
obs_T[1] * ihdp_separated[1].shape[0],
obs_T[2] * ihdp_separated[2].shape[0],
obs_T[3] * ihdp_separated[3].shape[0],
obs_T[4] * ihdp_separated[4].shape[0]
]).T

exp_failures = np.array([
(1 - obs_T[0]) * ihdp_separated[0].shape[0],
(1 - obs_T[1]) * ihdp_separated[1].shape[0],
(1 - obs_T[2]) * ihdp_separated[2].shape[0],
(1 - obs_T[3]) * ihdp_separated[3].shape[0],
(1 - obs_T[4]) * ihdp_separated[4].shape[0]
]).T

exp_successes = np.sum(exp_successes, axis=1).astype(int)
exp_failures = np.sum(exp_failures, axis=1).astype(int)
exp_successes
exp_failures

#%%
np.save('ihdp_5_payout', obs)
np.save('ihdp_5_obs_success', successes)
np.save('ihdp_5_obs_failure', failures)
np.save('ihdp_5_exp_p', exp)
np.save('ihdp_5_exp_success', exp_successes)
np.save('ihdp_5_exp_failure', exp_failures)

#%% SAVE IN EXISTING FORMAT

# Check existing format
data_exp = np.load('data_exp.npy')
data_obs = np.load('data_obs.npy')

data_exp
data_obs

data_exp_success = np.load('ihdp_exp_success.npy')
data_exp_failure = np.load('ihdp_exp_failure.npy')
data_exp_ihdp = np.vstack((data_exp_failure, data_exp_success)).T
data_exp_ihdp
np.save('data_exp_ihdp', data_exp_ihdp)

data_obs_success = np.load('ihdp_obs_success.npy')
data_obs_failure = np.load('ihdp_obs_failure.npy')
data_obs_ihdp = np.vstack((data_obs_failure, data_obs_success)).T
data_obs_ihdp
np.save('data_obs_ihdp', data_obs_ihdp)

# %% LOAD DATA
THETA = np.load('ihdp_payout.npy')
THETA.round(3)

data_exp = np.load('data_exp_ihdp.npy')
data_obs = np.load('data_obs_ihdp.npy')
data_exp
data_obs

#%%
data_exp = np.load('data_exp.npy')
data_obs = np.load('data_obs.npy')
data_exp
data_obs
THETA = np.array([[0.2, 0.3, 0.5, 0.6],
               [0.6, 0.2, 0.3, 0.5],
               [0.5, 0.6, 0.2, 0.3],
               [0.3, 0.5, 0.6, 0.2]])

#%% FIVE ACTIONS
# data_obs_success = np.load('ihdp_5_obs_success.npy')
# data_obs_failure = np.load('ihdp_5_obs_failure.npy')
# data_obs_ihdp = np.vstack((data_obs_failure, data_obs_success)).T
# data_obs_ihdp
# np.save('data_obs_ihdp_5', data_obs_ihdp)
#
# data_exp_success = np.load('ihdp_5_exp_success.npy')
# data_exp_failure = np.load('ihdp_5_exp_failure.npy')
# data_exp_ihdp = np.vstack((data_exp_failure, data_exp_success)).T
# data_exp_ihdp
# np.save('data_exp_ihdp_5', data_exp_ihdp)

#%% TEST

# Observational data
E_obs = data_obs[:, 1] / (data_obs[:, 0] + data_obs[:, 1])     # E_obs(Y|X)
p_int = np.sum(data_obs, axis=1) / np.sum(data_obs)  # P(I) from observational data, X=intent

# Experimental data
E_exp = data_exp[:, 1] / (data_exp[:, 0] + data_exp[:, 1])     # E_exp(Y|do(X))
experimental_successes = data_exp[:, 1]
experimental_failures = data_exp[:, 0]
N_data_exp = np.sum(data_exp, axis=1)  # Total experimental data for each action

E_obs
p_int
E_exp
experimental_successes
experimental_failures
N_data_exp
THETA


#%% CREATE I SAMPLES
config.K
T = 100
np.random.randint(config.K, size=T)

theta_estimate = np.array([[0.2, 0.3, 0.5, 0.6],
                       [0.6, 0.2, 0.3, 0.5],
                       [0.5, 0.6, 0.2, 0.3],
                       [0.3, 0.5, 0.6, 0.2]])

THETA
np.abs(THETA - theta_estimate)

np.mean(np.abs(THETA - theta_estimate) / THETA)

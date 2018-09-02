"""
MABUC (Bayesian) run file
"""

import numpy as np
import pymc3 as pm
# from numpy.linalg import inv
# from scipy.stats import beta
# import matplotlib.pyplot as plt
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

data_exp = np.load('data_exp.npy')
data_obs = np.load('data_obs.npy')

E_obs = data_obs[:, 1] / (data_obs[:, 0] + data_obs[:, 1])     # E_obs(Y|X)
E_exp = data_exp[:, 1] / (data_exp[:, 0] + data_exp[:, 1])     # E_exp(Y|do(X))
# P(X) from observational data, X=intent
p_int = np.sum(data_obs, axis=1) / np.sum(data_obs)

assert (E_obs <= 1).all()
assert (E_obs >= 0).all()
assert (E_exp <= 1).all()
assert (E_exp >= 0).all()
assert (p_int <= 1).all()
assert (p_int >= 0).all()


class MabucBayesianAlgorithm:
    '''
    Bayesian version - sampling
    '''

    def __init__(self, algorithm_name):
        self.algorithm_name = algorithm_name

    def run_simulation(self):
        """
        Run one simulation for T timesteps (T exploration steps)
        """

        # Instantiate (sample) confounders for all T steps
        U_samples = np.random.randint(2, size=(config.N_CONFOUNDERS, config.T))
        I_samples = np.array(config.intent(U_samples[0], U_samples[1]))

        # ------------------------ #
        # Initialize data matrix   #
        # ------------------------ #
        # successes (4 arms x 4 Intents)
        self.s = np.ones((config.K, config.K))
        self.f = np.ones((config.K, config.K))  # failures

        # Exploration + observation data matrix
        self.s_with_obs = (self.s + np.diag(data_obs[:, 1])).astype(int)
        self.f_with_obs = (self.f + np.diag(data_obs[:, 0])).astype(int)

        # Record results
        Action = np.zeros(config.T)  # Action (0-3) taken for each timestep
        Reward = np.zeros(config.T)  # Reward (0,1) for each timestep
        Prob = np.zeros(config.T)  # Best action or not (0,1) for each timestep
        Conds = np.zeros(config.U)  # Confounder setting count

        # BAYESIAN POSTERIOR (initialise to observation)
        self.N_data = (self.s_with_obs + self.f_with_obs).astype(int)
        self.p_wins = self.s_with_obs / self.N_data
        self.y_post = self.p_wins

        self.exploration_count = np.zeros((config.K, config.K))

        # Execute exploration for each timestep t
        #batch_counter = 0

        for t in range(config.T):

            self.I = I_samples[t]
            covariateIndex = self.I
            Conds[covariateIndex] += 1

            #----------------------------#
            # Update expected rewards P  #
            #----------------------------#
            # Choose action
            if self.algorithm_name in ['bayesian']:

                #if batch_counter == config.N_BATCH:
                if t % config.N_BATCH == 0:
                    # Update model and get new posterior
                    # Choose action from posterior
                    print('Obtaining posterior, t = {} \n'.format(t))
                    self.y_post, model, trace = self.get_posterior(self.p_wins)
                    choices = self.y_post[:, self.I]
                    action = np.argmax(choices)
                    #batch_counter = 0  # reset batch counter
                    self.exploration_count[action, self.I] += 1
                    print("Exploration count = {} \n".format(self.exploration_count))
                    print("y_post = {} \n".format(self.y_post))

                else:
                    # Choose random action:
                    # choices = np.delete(np.arange(config.K), self.I)
                    # action = np.random.choice(choices)

                    # Or choose best apparent action:
                    # choices = self.p_wins[:, self.I]
                    # action = np.argmax(choices)

                    # Or sample from available model
                    ppc_sample = pm.sample_ppc(trace, samples=1, model=model)
                    # sample = np.mean(ppc_sample['L'], axis=(0, 1))  # BERNOULLI
                    sample = (np.mean(ppc_sample['L'], axis=0) / self.N_data)  # BINOMIAL
                    choices = sample[:, self.I]
                    action = np.argmax(choices)
                    self.exploration_count[action, self.I] += 1

            # Probability of success
            win_prob = config.THETA[action, covariateIndex]

            # Pull arm
            reward = np.random.choice(2, p=[1 - win_prob, win_prob])

            # # Update collected data matrix
            # self.s[action, self.I] += reward
            # self.f[action, self.I] += 1 - reward

            self.s_with_obs[action, self.I] += reward
            self.f_with_obs[action, self.I] += 1 - reward

            # Payout based on data only
            self.N_data = (self.s_with_obs + self.f_with_obs).astype(int)
            self.p_wins = self.s_with_obs / self.N_data

            # Record
            Action[t] = action
            Reward[t] = reward
            # [bestVal, bestAction] = max(theta(:, covariateIndex));
            bestAction = np.argmax(config.THETA[:, covariateIndex])
            #bestVal = config.THETA[bestAction, covariateIndex] # Not used
            Prob[t] = 1 if action == bestAction else 0

            #batch_counter += 1

        return Action, Reward, Prob, Conds

    def get_posterior(self, observed_data, use_ppc_samples=config.USE_PPC_SAMPLES):

        # ------------------------------------------------- #
        # CREATE BERNOULLI INPUT FROM SUCCESS PROPORTION    #
        # ------------------------------------------------- #
        # Creates N_bernoulli_sims lots of (K x K) boolean grid mimicking
        # success rate in observed data. Required because pm.Bernoulli only
        # takes boolean data shaped (N x K x K) (this method is quite hacky)
        N_bernoulli_sims = 500
        data = np.ones((config.K, config.K, N_bernoulli_sims))
        change_to_zeros = np.round((1 - self.p_wins) * N_bernoulli_sims).astype(int)
        data[change_to_zeros[:, :, None] > np.arange(data.shape[-1])] = 0
        data = np.transpose(data, (2, 0, 1))  # Swap axes
        data = np.take(data, np.random.rand(data.shape[0]).argsort(), axis=0)  # Shuffle on N axis

        # ------------------------------------------------- #
        # USE PYMC3 TO CREATE POSTERIOR AND SAMPLE FROM IT  #
        # ------------------------------------------------- #
        model = pm.Model()
        with model:

            # Priors for unknown model parameters
            a = pm.Normal('a', mu=0, sd=10, shape=(config.K, 1))
            b = pm.Normal('b', mu=0, sd=10, shape=(1, config.K))
            offset = pm.Normal('offset', mu=0, sd=10)

            p = pm.Deterministic('p', self.sigmoid(a + b + offset))

            # Likelihood (sampling distribution) of observations
            # L = pm.Bernoulli('L', p=p, observed=data)
            L = pm.Binomial('L', self.N_data, p, observed=self.s_with_obs)

            # draw posterior samples
            trace = pm.sample(config.TRACE_LENGTH, nuts_kwargs=dict(target_accept=.95), chains=config.N_MCMC_CHAINS)

        if use_ppc_samples:
            # Use samples from ppc to obtain posterior point estimates
            ppc = pm.sample_ppc(trace, samples=config.N_PPC_SAMPLES, model=model)
            # y_post = np.mean(ppc['L'], axis=(0, 1))  # USE IF USING BERNOULLI
            y_post = np.mean(ppc['L'], axis=0) / self.N_data  # USE IF USING BINOMIAL

        else:
            # Use trace to obtain posterior point estimates
            a_post = np.array(np.mean(trace[:100]['a'], axis=0))
            b_post = np.array(np.mean(trace[:100]['b'], axis=0))
            offset_post = np.mean(trace[:100]['offset'], axis=0)
            y_post = self.sigmoid(a_post + b_post + offset_post)

        return y_post, model, trace

    @staticmethod
    def sigmoid(x):
        """ Numerically stable sigmoid function """
        z = np.exp(x)
        return z / (1 + z)

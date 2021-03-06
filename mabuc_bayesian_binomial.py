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


class MabucBayesianAlgorithm:
    '''
    Bayesian version - sampling
    '''

    def __init__(self, algorithm_name):
        self.algorithm_name = algorithm_name

    def run_simulation(self, n):
        """
        Run one simulation for T timesteps (T exploration steps)
        """

        self.n = n

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

                if t % config.N_BATCH == 0:
                    # Update model and get new posterior
                    # Choose action from posterior
                    print('Obtaining posterior, N = {}, t = {} \n'.format(self.n, t))
                    self.y_post, model, trace = self.get_posterior()
                    choices = self.y_post[:, self.I]
                    action = np.argmax(choices)

                    # Record and print information
                    self.exploration_count[action, self.I] += 1
                    print("N = {}, Exploration count = \n {} \n".format(self.n, self.exploration_count))
                    print("N = {}, y_post = \n {} \n".format(self.n, self.y_post.round(3)))

                else:
                    # Choose random action:
                    # choices = np.delete(np.arange(config.K), self.I)
                    # action = np.random.choice(choices)

                    # Or choose best apparent action:
                    # choices = self.p_wins[:, self.I]
                    # action = np.argmax(choices)

                    # Or sample from available model
                    ppc_sample = pm.sample_ppc(trace, samples=1, model=model)
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

    def get_posterior(self):

        # ------------------------------------------------- #
        # USE PYMC3 TO CREATE POSTERIOR AND SAMPLE FROM IT  #
        # ------------------------------------------------- #
        model = pm.Model()
        with model:

            # Priors for unknown model parameters
            hyper_mu = pm.Normal('hyper_mu', mu=0, sd=10)
            hyper_sd = pm.Gamma('hyper_sd', alpha=config.ALPHA_HYPER_GAMMA_SD,
                                beta=config.BETA_HYPER_GAMMA_SD)
            c = pm.Normal('c', mu=hyper_mu, sd=hyper_sd, shape=(config.K, config.K))

            # Expected success rate
            p = config.sigmoid(c)

            # Likelihood (sampling distribution) of observations
            pm.Binomial('L', n=self.N_data, p=p, observed=self.s_with_obs)

            # Draw posterior samples
            trace = pm.sample(config.TRACE_LENGTH, nuts_kwargs=dict(target_accept=.9,
                              max_treedepth=20), chains=config.N_MCMC_CHAINS)

        if config.USE_PPC_SAMPLES:
            # Use samples from ppc to obtain posterior point estimates
            ppc = pm.sample_ppc(trace, samples=config.N_PPC_SAMPLES, model=model)
            y_post = np.mean(ppc['L'], axis=0) / self.N_data  # USE IF USING BINOMIAL

        else:
            # Use trace to obtain posterior point estimates
            c_ = np.array(np.mean(trace[:(config.TRACE_LENGTH / 2)]['c'], axis=0))
            y_post = self.sigmoid(c_)

        return y_post, model, trace

    @staticmethod
    def sigmoid(x):
        """ Numerically stable sigmoid function """
        z = np.exp(x)
        return z / (1 + z)

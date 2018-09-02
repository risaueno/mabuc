"""
MABUC (Bayesian) run file
"""

import numpy as np
import pymc3 as pm
import theano.tensor as tt
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
        self.observed_successes = (self.s + np.diag(data_obs[:, 1])).astype(int)
        self.observed_failures = (self.f + np.diag(data_obs[:, 0])).astype(int)

        # Record results
        Action = np.zeros(config.T)  # Action (0-3) taken for each timestep
        Reward = np.zeros(config.T)  # Reward (0,1) for each timestep
        Prob = np.zeros(config.T)  # Best action or not (0,1) for each timestep
        Conds = np.zeros(config.U)  # Confounder setting count

        # BAYESIAN POSTERIOR (initialise to observation)
        self.N_data = (self.observed_successes + self.observed_failures).astype(int)
        self.p_wins = self.observed_successes / self.N_data
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
            if self.algorithm_name in ['MCMC']:

                if t % config.N_BATCH == 0:
                    # Update model and get new posterior
                    # Choose action from posterior
                    print('Obtaining posterior, N = {}, t = {} \n'.format(self.n + 1, t))
                    self.y_post, model, trace = self.get_posterior()
                    choices = self.y_post[:, self.I]
                    action = np.argmax(choices)

                    # Record and print information
                    self.exploration_count[action, self.I] += 1
                    print("N = {}, Exploration count = \n {} \n".format(self.n + 1, self.exploration_count))
                    print("N = {}, y_post = \n {} \n".format(self.n + 1, self.y_post.round(3)))
                    counter = 1

                else:
                    # Choose random action:
                    # choices = np.delete(np.arange(config.K), self.I)
                    # action = np.random.choice(choices)

                    # Or choose best apparent action:
                    # choices = self.p_wins[:, self.I]
                    # action = np.argmax(choices)

                    # Or sample from available model
                    # ppc_sample = pm.sample_ppc(trace, samples=1, model=model)
                    # sample = (np.mean(ppc_sample['L'], axis=0) / self.N_data)  # BINOMIAL
                    # choices = sample[:, self.I]
                    # action = np.argmax(choices)

                    # Or get sample from trace
                    #print(trace[-counter]['c'])
                    #sample = config.sigmoid(np.mean(trace[-counter]['c'], axis=0))
                    # print(trace[-counter]['c'])

                    if config.USE_MODEL_WITH_CONSTRAINT:
                        a_post = trace[-counter]['a']
                        b_post = trace[-counter]['b']
                        intercept_post = trace[-counter]['offset']
                        sample = config.sigmoid(a_post + b_post + intercept_post)
                        action = np.argmax(choices)
                    else:
                        sample = config.sigmoid(trace[-counter]['c'])
                        choices = sample[:, self.I]
                        action = np.argmax(choices)

                    self.exploration_count[action, self.I] += 1
                    counter += 1

            # Probability of success
            win_prob = config.THETA[action, covariateIndex]

            # Pull arm
            reward = np.random.choice(2, p=[1 - win_prob, win_prob])

            # # Update collected data matrix
            # self.s[action, self.I] += reward
            # self.f[action, self.I] += 1 - reward

            self.observed_successes[action, self.I] += reward
            self.observed_failures[action, self.I] += 1 - reward

            # Payout based on data only
            self.N_data = (self.observed_successes + self.observed_failures).astype(int)
            self.p_wins = self.observed_successes / self.N_data

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

            if config.USE_MODEL_WITH_CONSTRAINT:
                # Priors for unknown model parameters
                hyper_mu = pm.Normal('hyper_mu', mu=0, sd=10)
                hyper_sd = pm.Gamma('hyper_sd', alpha=config.ALPHA_HYPER_GAMMA_SD,
                                    beta=config.BETA_HYPER_GAMMA_SD)
                a = pm.Normal('a', mu=hyper_mu, sd=hyper_sd, shape=(config.K, 1))
                b = pm.Normal('b', mu=hyper_mu, sd=hyper_sd, shape=(1, config.K))
                offset = pm.Normal('offset', mu=hyper_mu, sd=hyper_sd)

                c = a + b + offset
                p = config.sigmoid(c)

            else:
                # Priors for unknown model parameters
                hyper_mu = pm.Normal('hyper_mu', mu=0, sd=10)
                hyper_sd = pm.Gamma('hyper_sd', alpha=config.ALPHA_HYPER_GAMMA_SD,
                                    beta=config.BETA_HYPER_GAMMA_SD)
                c = pm.Normal('c', mu=hyper_mu, sd=hyper_sd, shape=(config.K, config.K))

                p = config.sigmoid(c)

            # Binomial part
            gamma = pm.Potential('gamma', self.observed_successes * tt.log(p) + self.observed_failures * tt.log(1 - p))

            # Nuisance parameters for P(I)
            # theta = pm.Dirichlet('theta', E_exp, shape=(1, 4))
            theta = pm.Dirichlet('theta', data_exp[:, 0], shape=(1, 4))

            # Joint distribution P(Y, I, A) is proportional to:
            def joint(gamma, theta):
                return gamma * theta

            # Likelihood (sampling distribution) of observations (custom)
            pm.DensityDist('L', joint, observed={'gamma': gamma, 'theta': theta})

            # Draw posterior samples
            trace = pm.sample(config.TRACE_LENGTH, nuts_kwargs=dict(target_accept=.9,
                              max_treedepth=20), chains=config.N_MCMC_CHAINS)

        if config.USE_PPC_SAMPLES:
            # Use samples from ppc to obtain posterior point estimates
            ppc = pm.sample_ppc(trace, samples=config.N_PPC_SAMPLES, model=model)
            y_post = np.mean(ppc['L'], axis=0) / self.N_data  # USE IF USING BINOMIAL

        else:
            # Use trace to obtain posterior point estimates
            c_ = np.array(np.mean(trace[:config.N_TRACE_SAMPLES]['c'], axis=0))
            y_post = self.sigmoid(c_)

        return y_post, model, trace

    @staticmethod
    def sigmoid(x):
        """ Numerically stable sigmoid function """
        z = np.exp(x)
        return z / (1 + z)

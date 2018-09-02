"""
MABUC run file
"""

import numpy as np
# import pymc3 as pm
# from numpy.linalg import inv
from scipy.stats import beta
# import matplotlib.pyplot as plt
# import math
# from collections import defaultdict
# import sys
# import os
# import random

from config import Config
config = Config()


class MabucAlgorithm:
    '''
    MABUC using inverse variance weighted average
    '''

    def __init__(self, algorithm_name):
        self.algorithm_name = algorithm_name
        self.USE_INTERVENTIONAL_DATA = True if '++' in algorithm_name else False
        self.USE_OBSERVATIONAL_DATA = True if '+' in algorithm_name else False
        self.USE_WEIGHTED_TS = True if 'original' in algorithm_name else False      # Original matlab implementation - weird
        self.USE_INV_VAR_IN_XINT = True if 'invvar' in algorithm_name else False

    def run_simulation(self, n, T, I_samples, data_lists):
        """
        Run one simulation for T timesteps (T exploration steps)
        """
        #print('Loading data... \n')

        self.n = n

        if config.USE_RANDOM_DATA:
            data_exp_list = data_lists[0]
            data_obs_list = data_lists[1]
            theta_list = data_lists[2]

            theta = theta_list[n]
            data_exp = data_exp_list[n]
            data_obs = data_obs_list[n]

            self.p_int = np.sum(data_obs, axis=1) / np.sum(data_obs)  # P(I) from observational data, X=intent
            self.E_exp = data_exp[:, 1] / (data_exp[:, 0] + data_exp[:, 1])     # E_exp(Y|do(X))
            # experimental_successes = data_exp[:, 1]
            # experimental_failures = data_exp[:, 0]
            # N_data_exp = np.sum(data_exp, axis=1)  # Total experimental data for each action

        else:
            theta = config.THETA
            data_exp = config.data_exp
            data_obs = config.data_obs

            self.p_int = np.sum(data_obs, axis=1) / np.sum(data_obs)  # P(I) from observational data, X=intent
            self.E_exp = data_exp[:, 1] / (data_exp[:, 0] + data_exp[:, 1])     # E_exp(Y|do(X))

        # Let config.T overwrite T. For now set config.N_BATCH to be config.N
        if config.USE_EXISTING_I_SAMPLES:
            I_samples = I_samples[:config.T]

        # Initialize exploration data matrix
        # successes (4 arms x 4 Intents)
        self.s = np.ones((config.K, config.K))
        self.f = np.ones((config.K, config.K))  # failures

        # Exploration + observation data matrix
        self.s_with_obs = self.s + np.diag(data_obs[:, 1])
        self.f_with_obs = self.f + np.diag(data_obs[:, 0])
        self.p_wins = self.s_with_obs / (self.s_with_obs + self.f_with_obs)

        # Record results
        Intent = np.zeros(config.T)
        Action = np.zeros(config.T)  # Action (0-3) taken for each timestep
        Reward = np.zeros(config.T)  # Reward (0,1) for each timestep
        Prob = np.zeros(config.T)  # Best action or not (0,1) for each timestep
        # Conds = np.zeros(config.U)  # Confounder setting count

        CumProb = np.zeros(config.T)
        PayoutEstimate = self.p_wins
        AveragePayoutAccuracy = np.zeros(config.T)

        # print('Loaded data. Activating agent... \n')

        # Execute one timestep (exploration)
        for t in range(config.T):

            self.I = I_samples[t].astype(int)

            E_comb, V_comb = self.get_combined_estimate()

            if self.USE_WEIGHTED_TS:
                currentWeights = E_comb
                choices = np.random.beta(self.s[:, self.I], self.f[:, self.I]) * currentWeights  # original # Why weigh with beta sampling here??
            else:
                choices = E_comb

            # Choose best action from choices
            action = np.argmax(choices)

            # Probability of success
            win_prob = config.THETA[action, self.I]

            # Pull arm
            reward = np.random.choice(2, p=[1 - win_prob, win_prob])

            # Update collected data matrix
            self.s[action, self.I] += reward
            self.f[action, self.I] += 1 - reward
            self.s_with_obs[action, self.I] += reward
            self.f_with_obs[action, self.I] += 1 - reward
            self.p_wins = self.s_with_obs / (self.s_with_obs + self.f_with_obs)

            # Record
            Intent[t] = self.I
            Action[t] = action
            Reward[t] = reward
            # bestVal = config.THETA[bestAction, self.I]
            PayoutEstimate[:, self.I] = E_comb
            AveragePayoutAccuracy[t] = 1 - np.mean(np.abs(PayoutEstimate - theta) / theta)

            bestAction = np.argmax(theta[:, self.I])
            if action == bestAction:
                Prob[t] = 1
                CumProb[t] = CumProb[t - 1] + 1
            else:
                Prob[t] = 0
                CumProb[t] = CumProb[t - 1]

        CumProb = CumProb / np.arange(1, config.T + 1)

        return Intent, Action, Reward, Prob, CumProb, AveragePayoutAccuracy
        # return Action, Reward, Cumulative_Prob, Conds

    def get_combined_estimate(self):
        '''
        Get combined payout estimates (sample and XInt combined through inverse variance weighted average)
        for given intent I
        '''

        # totals = self.s[:, self.I] + self.f[:, self.I]  # Total samples under this intent

        #------------------------#
        # Get E_samp and V_samp  #
        #------------------------#
        # 'IVWA*_edit_', 'IVWA*_edit': NO SAMPLING, JUST DATA
        # 'IVWA_paper': SAMPLE HERE FROM EXPLORATION ONLY
        # 'IVWA*_2': SAMPLE HERE FROM EXPL + OBS DATA

        l = ['IVWA - NO TS']  # NO TS!
        # if self.algorithm_name in :
        if self.algorithm_name.startswith(tuple(l)):
            if self.USE_OBSERVATIONAL_DATA:
                E_samp, V_samp = beta.stats(self.s_with_obs[:, self.I], self.f_with_obs[:, self.I])  # (3) CHANGE
            else:
                E_samp, V_samp = beta.stats(self.s[:, self.I], self.f[:, self.I])

        # 'ORIGINAL' - NO TS SAMPLE HERE
        if self.USE_WEIGHTED_TS:
            E_samp, V_samp = beta.stats(self.s[:, self.I], self.f[:, self.I])

        # TS SAMPLE from experiment AND obs successes and failures
        l = ['IVWA - TS']
        if self.algorithm_name.startswith(tuple(l)):
            if self.USE_OBSERVATIONAL_DATA:
                _, V_samp = beta.stats(self.s_with_obs[:, self.I], self.f_with_obs[:, self.I])
                E_samp = np.random.beta(self.s_with_obs[:, self.I], self.f_with_obs[:, self.I])  # TS SAMPLE HERE
            else:
                _, V_samp = beta.stats(self.s[:, self.I], self.f[:, self.I])
                E_samp = np.random.beta(self.s[:, self.I], self.f[:, self.I])  # TS SAMPLE HERE

        # TS sample from experiment successes and failures
        l = ['IVWA - paper', 'IVWA - editTS']
        if self.algorithm_name.startswith(tuple(l)):
            _, V_samp = beta.stats(self.s[:, self.I], self.f[:, self.I])
            E_samp = np.random.beta(self.s[:, self.I], self.f[:, self.I])  # TS SAMPLE HERE

        #------------------------#
        # Get E_XInt and V_XInt  #
        #------------------------#
        # XINT, XARM ONLY POSSIBLE WITH EXPERIMENTAL DATA??

        if self.USE_INTERVENTIONAL_DATA:
            E_xint, V_xint = self.get_xint()

        #------------------------#
        # Get E_Comb             #
        #------------------------#

        if self.USE_INTERVENTIONAL_DATA:
            M = np.array([[E_samp, V_samp], [E_xint, V_xint]])  # E_samp etc: K
            E_comb, V_comb = self.inverse_variance_weighting(M)

        else:
            E_comb = E_samp
            V_comb = V_samp

        return E_comb, V_comb

    def get_xint(self):
        '''
        Given intent I get cross intentional payout estimates for all X arms
        '''

        # Cross intentional variance: use observations as well (averaged variance will be smaller, more reliable than SAMP)

        # Compute variances

        l = ['IVWA - paper', 'IVWA - NO TS', 'IVWA - TS']
        # if self.algorithm_name in :
        if self.algorithm_name.startswith(tuple(l)):
            if self.USE_OBSERVATIONAL_DATA:
                _, variances = beta.stats(self.s_with_obs, self.f_with_obs)
            else:
                _, variances = beta.stats(self.s, self.f)

        l = ['IVWA - original', 'IVWA - editTS']
        # if self.algorithm_name in ['IVWA_original++']:
        if self.algorithm_name.startswith(tuple(l)):
            _, variances = beta.stats(self.s, self.f)

        # l = ['IVWA*_edit']
        # # if self.algorithm_name in :
        # if self.algorithm_name.startswith(tuple(l)):
        if self.USE_INV_VAR_IN_XINT:
            V_xint = 1 / (np.sum(np.delete(1 / variances, self.I, axis=1), axis=1) / (config.K - 1))
        else:
            # THE VARIENCE HERE IS AVERAGED - IS THIS RIGHT (says so in paper - seems to work better than inverse averaging...)
            V_xint = np.sum(np.delete(variances, self.I, axis=1), axis=1) / (config.K - 1)

        # Compute payout estimates
        # p_wins incorporates both observation and
        # E_xint = np.array([(E_exp[x] - np.sum(np.delete(self.p_wins[x, :] * p_int, self.I))) / p_int[self.I]
        #                    for x in range(config.K)])  # (2) CHANGE - USED LIST COMPREHENSION
        E_xint = np.array([(self.E_exp[x] - np.sum(np.delete(self.p_wins[x, :] * self.p_int, self.I))) / self.p_int[self.I]
                           for x in range(config.K)])

        return E_xint, V_xint

    def inverse_variance_weighting(self, M):
        """
        Input: (inputs,2) matrix of rows formatted as [mean, variance]
        Returns: E_combo[Y(X)|I], Var_combo[Y(X)|I] for all X
        """
        # E_comb = np.sum(M[:,0] / M[:,1]) / np.sum(1 / M[:,1])

        E_comb = np.sum(M[:, 0, :] / M[:, 1, :], axis=0) / \
            np.sum(1 / M[:, 1, :], axis=0)
        #precision = 1/M[:,1,:]
        #precision_comb = np.mean

        # V_comb douse tsukawanai!

        l = ['*IVWA']
        # if self.algorithm_name in :
        # if self.algorithm_name in ['IVWA*_1', 'IVWA*_edit', 'IVWA*_edit_', 'IVWA*_edit_precision', 'IVWA*_edit_precision2']:
        if self.algorithm_name.startswith(tuple(l)):
            V_comb = 1 / (np.mean(1 / M[:, 1, :], axis=0))
        else:
            V_comb = np.mean(M[:, 1, :], axis=0)

        return E_comb, V_comb

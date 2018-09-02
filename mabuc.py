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

# if not config.USE_RANDOM_DATA:
# Load experimental and observational data
data_exp = config.data_exp
data_obs = config.data_obs

E_obs = config.E_obs     # E_obs(Y|X)
# E_exp = config.E_exp     # E_exp(Y|do(X))
# p_int = config.p_int     # P(I) from observational data


class MabucAlgorithm:
    '''
    MABUC using inverse variance weighted average
    '''

    def __init__(self, algorithm_name):
        self.algorithm_name = algorithm_name
        self.USE_INTERVENTIONAL_DATA = True if '++' in algorithm_name else False
        self.USE_OBSERVATIONAL_DATA = True if '+' in algorithm_name else False

    def run_simulation(self, n, T, I_samples, data_lists=None):
        """
        Run one simulation for T timesteps (T exploration steps)
        """
        self.n = n

        if config.USE_RANDOM_DATA:
            data_exp_list = data_lists[0]
            data_obs_list = data_lists[1]
            theta_list = data_lists[2]

            theta = theta_list[n]
            data_exp = data_exp_list[n]
            data_obs = data_obs_list[n]
            # Observational data
            # E_obs = data_obs[:, 1] / (data_obs[:, 0] + data_obs[:, 1])     # E_obs(Y|X)
            self.p_int = np.sum(data_obs, axis=1) / np.sum(data_obs)  # P(I) from observational data, X=intent

            # Experimental data
            self.E_exp = data_exp[:, 1] / (data_exp[:, 0] + data_exp[:, 1])     # E_exp(Y|do(X))
            # experimental_successes = data_exp[:, 1]
            # experimental_failures = data_exp[:, 0]
            # N_data_exp = np.sum(data_exp, axis=1)  # Total experimental data for each action

        else:
            theta = config.THETA
            data_exp = config.data_exp
            data_obs = config.data_obs
            # E_obs = config.E_obs     # E_obs(Y|X)
            self.E_exp = config.E_exp     # E_exp(Y|do(X))
            self.p_int = config.p_int     # P(I) from observational data

        # Let config.T overwrite T. For now set config.N_BATCH to be config.N
        if config.USE_EXISTING_I_SAMPLES:
            I_samples = I_samples[:config.T]

        # # Instantiate (sample) confounders for all T steps
        # U_samples = np.random.randint(2, size=(config.N_CONFOUNDERS, config.T))
        # I_samples = np.array(config.intent(U_samples[0], U_samples[1]))

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
        Action = np.zeros(config.T)     # Action (0-3) taken for each timestep
        Reward = np.zeros(config.T)     # Reward (0,1) for each timestep
        Prob = np.zeros(config.T)       # Best action or not (0,1) for each timestep
        PayoutEstimate = self.p_wins
        AveragePayoutAccuracy = np.zeros(config.T)

        CumProb = np.zeros(config.T)
        PayoutEstimate = np.zeros((config.K, config.K))

        # Execute one timestep (exploration)
        for t in range(config.T):
            # I = GetIntent(U_samples[0,t], U_samples[1,t]) #intentEqn
            self.I = I_samples[t].astype(int)
            #self.I = config.intent(random.getrandbits(1), random.getrandbits(1))
            covariateIndex = self.I  # covariateIndexEqn
            # Conds[covariateIndex] += 1

            # We'll compute the weighted factors with aid from other intent conditions
            #_, _, wETT = self.get_combined_estimate()
            E_comb, V_comb = self.get_combined_estimate()

            # Get the weighted values
            # E_comb, vector of E[Y(actions)|I] computed by inv-weighting approach

            # Choose action
            # RDC1 originally here (directly below). No point maybe in not TS after, when Ecombo variance is CORRECTED
            if self.algorithm_name in ['RDC_paper', 'RDC_paper_NO_TS', 'RDC*_2', 'RDC*_edit', 'RDC*_edit_', 'RDC*_edit_NO_TS_3', 'RDC*_edit_TS_4', 'RDC*_edit_precision']:
                choices = E_comb
            if self.algorithm_name in ['RDC_original++']:
                currentWeights = E_comb
                choices = np.random.beta(self.s[:, self.I], self.f[:, self.I]) * currentWeights  # original
                # Why beta sampling here??
            if self.algorithm_name in ['RDC_test', 'RDC*_1', 'RDC*_edit_TS', 'RDC*_edit_TS_2', 'RDC*_edit_precision2']:
                # TS METHOD BY MODELLING AS BETA DISTRIBUTION FROM E_COMB AND V_COMB
                # THIS DOES NOT WORK VERY WELL!
                #print(E_comb, V_comb)
                a = ((1 - E_comb) / V_comb - (1 / E_comb)) * (E_comb ** 2)
                b = a * (1 / E_comb - 1)

                try:
                    assert all(a >= 0)
                    assert all(b >= 0)
                except:
                    #print('WARN: a = {} \n'.format(a))
                    #print('WARN: b = {} \n'.format(b))
                    #print('E_comb = {} \n, V_comb = {}'.format(E_comb, V_comb))

                    # ??? BETTER WAY ???
                    # (5) CHANGE - When a and b are negative, approximate using given data.
                    # TS sample at the end.
                    a[np.where(a <= 0)] = self.s_with_obs[np.where(a <= 0), self.I]
                    b[np.where(a <= 0)] = self.f_with_obs[np.where(a <= 0), self.I]

                    a[np.where(b <= 0)] = self.s_with_obs[np.where(b <= 0), self.I]
                    b[np.where(b <= 0)] = self.f_with_obs[np.where(b <= 0), self.I]

                    #print('WARN: a = {} \n'.format(a))
                    #print('WARN: b = {} \n'.format(b))

                choices = np.random.beta(a, b)

            action = np.argmax(choices)
            # maxVal = choices[action]  # Not used?

            # Probability of success
            win_prob = theta[action, covariateIndex]

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
            PayoutEstimate[:, self.I] = E_comb
            AveragePayoutAccuracy[t] = 1 - np.mean(np.abs(PayoutEstimate - theta) / theta)

            bestAction = np.argmax(theta[:, covariateIndex])
            if action == bestAction:
                Prob[t] = 1
                CumProb[t] = CumProb[t - 1] + 1
            else:
                Prob[t] = 0
                CumProb[t] = CumProb[t - 1]

        CumProb = CumProb / np.arange(1, config.T + 1)

        # return Intent, Action, Reward, Prob, CumProb, Conds
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
        if self.algorithm_name in ['RDC_test', 'RDC_paper_NO_TS', 'RDC*_1', 'RDC*_edit', 'RDC*_edit_', 'RDC*_edit_TS', 'RDC*_edit_TS_2', 'RDC*_edit_NO_TS_3', 'RDC*_edit_precision2']:
            E_samp, V_samp = beta.stats(
                self.s_with_obs[:, self.I], self.f_with_obs[:, self.I])
        if self.algorithm_name in ['RDC_original++']:
            E_samp, V_samp = beta.stats(self.s[:, self.I], self.f[:, self.I])

        if self.algorithm_name in ['RDC_test_2', 'RDC*_2', 'RDC*_edit_TS_4', 'RDC*_edit_precision']:
            _, V_samp = beta.stats(
                self.s_with_obs[:, self.I], self.f_with_obs[:, self.I])
            E_samp = np.random.beta(
                self.s_with_obs[:, self.I], self.f_with_obs[:, self.I])  # (6) CHANGE - TS SAMPLE HERE

        if self.algorithm_name in ['RDC_paper']:
            _, V_samp = beta.stats(
                self.s[:, self.I], self.f[:, self.I])
            E_samp = np.random.beta(
                self.s[:, self.I], self.f[:, self.I])  # (6) CHANGE - TS SAMPLE HERE

        #------------------------#
        # Get E_XInt and V_XInt  #
        #------------------------#
        E_xint, V_xint = self.get_xint()

        # try:
        #     assert all(E_xint <= 1)
        #     assert all(E_xint >= 0)
        # except:
        #     if self.algorithm_name in ['RDC*_edit_TS_2']:
        #         pass
        #     else:
        #         pass
            #print('WARN: E_xint = {}'.format(E_xint))
            #print('E_xint = {} \n p_wins = {} \n I = {}\n'.format(E_xint, self.p_wins, self.I))

        #m_xint = [1.,1.,1.,1.]; var_xint = [1.,1.,1.,1.];

        # Get E_XArm and V_XArm
        #m_xarm, var_xarm = self.Heuristic_XArm()
        #m_xarm = [1.,1.,1.,1.]; var_xarm = [1.,1.,1.,1.]

        #------------------------#
        # Get E_Comb             #
        #------------------------#
        M = np.array([[E_samp, V_samp], [E_xint, V_xint]])  # E_samp etc: K
        E_comb, V_comb = self.inverse_variance_weighting(M)

        # try:
        #     assert all(E_comb <= 1)
        #     assert all(E_comb >= 0)
        # except:
        #
        #     # If E_comb is negative, estimate from available data
        #     #print('\n WARN: E_comb = {}'.format(E_comb))
        #     if self.algorithm_name in ['RDC*_edit_TS_2', 'RDC*_edit_NO_TS_3']:
        #         E_comb[np.where(E_comb <= 0)] = self.p_wins[np.where(
        #             E_comb <= 0), self.I]
        #         #V_comb[np.where(E_comb <= 0)] = V_samp[np.where(E_comb <= 0)]
        #     else:
        #         pass
        #
        # try:
        #     assert all(V_comb >= 0)
        # except:
        #     print('\n WARN: V_comb = {}'.format(V_comb))

        # Compute weighted s and f values based on above
        #weightedS = totals * E_comb
        #weightedF = totals * (1 - E_comb)

        return E_comb, V_comb
        # return weightedS, weightedF, E_comb

    def get_xint(self):
        '''
        Given intent I get cross intentional payout estimates for all X arms
        '''
        # if config.USE_RANDOM_DATA:
        #     p_int = self.p_int
        #     E_exp = self.E_exp

        # Compute variances
        if self.algorithm_name in ['RDC*_1', 'RDC*_2', 'RDC*_edit', 'RDC*_edit_', 'RDC*_edit_TS', 'RDC*_edit_TS_2', 'RDC*_edit_NO_TS_3', 'RDC*_edit_TS_4', 'RDC*_edit_precision', 'RDC*_edit_precision2']:
            # (1) CHANGE - from (s, f)
            _, variances = beta.stats(self.s_with_obs, self.f_with_obs)
        if self.algorithm_name in ['RDC_original++', 'RDC_test', 'RDC_paper', 'RDC_paper_NO_TS']:
            _, variances = beta.stats(self.s, self.f)
            # BETTER: bc we are simply averaging them, if we have obs data the weight for xint will be smaller than samp

        if self.algorithm_name in ['RDC*_1', 'RDC*_edit_precision', 'RDC*_edit_precision2']:
            V_xint = 1 / (np.sum(np.delete(1 / variances, self.I,
                                           axis=1), axis=1) / (config.K - 1))
        else:
            # THE VARIENCE HERE IS AVERAGED - IS THIS RIGHT!?
            V_xint = np.sum(np.delete(variances, self.I, axis=1),
                            axis=1) / (config.K - 1)

        # Compute payout estimates
        E_xint = np.array([(self.E_exp[x] - np.sum(np.delete(self.p_wins[x, :] * self.p_int, self.I))) / self.p_int[self.I]
                           for x in range(config.K)])

        return E_xint, V_xint

    def inverse_variance_weighting(self, M):
        """
        Input: (N,2) matrix of rows formatted as [mean, variance]
        Returns: E_combo[Y(X)|I], Var_combo[Y(X)|I] for all X
        """
        # E_comb = np.sum(M[:,0] / M[:,1]) / np.sum(1 / M[:,1])

        E_comb = np.sum(M[:, 0, :] / M[:, 1, :], axis=0) / \
            np.sum(1 / M[:, 1, :], axis=0)
        #precision = 1/M[:,1,:]
        #precision_comb = np.mean
        if self.algorithm_name in ['RDC*_1', 'RDC*_edit_', 'RDC*_edit_precision', 'RDC*_edit_precision2']:
            V_comb = 1 / (np.mean(1 / M[:, 1, :], axis=0))
        else:
            V_comb = np.mean(M[:, 1, :], axis=0)

        return E_comb, V_comb

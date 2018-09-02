"""
MABUC (Bayesian) run file
"""

import numpy as np
import pymc3 as pm

from config import Config
config = Config()


class MabucBayesianAlgorithm:
    '''
    Bayesian version of MABUC - MCMC sampling of posterior parameters using Hamiltonial Monte Carlo.
    Algorithms:
    * 'MCMC++' = use counterfactual, observational and experimental data
    * 'MCMC+'  = use counterfactual and observational data only
    * 'MCMC'   = use counterfactual data only
    '''

    def __init__(self, algorithm_name, algorithm_index):
        self.algorithm_name = algorithm_name
        self.USE_INTERVENTIONAL_DATA = True if algorithm_name.startswith('MCMC++') else False
        self.USE_OBSERVATIONAL_DATA = True if algorithm_name.startswith('MCMC+') else False
        self.algorithm_index = algorithm_index

    def run_simulation(self, n, T, I_samples, data_lists=None):
        """
        Run one simulation for T timesteps (T exploration steps)
        """

        print('Loading data... \n')

        self.n = n

        #--------------------------------------------#
        # Load experimental and observational data   #
        #--------------------------------------------#

        if config.USE_RANDOM_DATA:
            data_exp_list = data_lists[0]
            data_obs_list = data_lists[1]
            theta_list = data_lists[2]

            self.theta = theta_list[n]
            data_exp = data_exp_list[n]
            data_obs = data_obs_list[n]

            # self.p_int = np.sum(data_obs, axis=1) / np.sum(data_obs)  # P(I) from observational data, X=intent
            # self.p_int_theta = self.p_int[:, np.newaxis].T
            #
            # # E_obs = data_obs[:, 1] / (data_obs[:, 0] + data_obs[:, 1])     # E_obs(Y|X)
            # self.E_exp = data_exp[:, 1] / (data_exp[:, 0] + data_exp[:, 1])     # E_exp(Y|do(X))
            # self.experimental_successes = data_exp[:, 1]
            # self.experimental_failures = data_exp[:, 0]
            # self.N_data_exp = np.sum(data_exp, axis=1)  # Total experimental data for each action

        else:
            self.theta = config.THETA
            data_exp = config.data_exp
            data_obs = config.data_obs

        self.p_int = np.sum(data_obs, axis=1) / np.sum(data_obs)  # P(I) from observational data, X=intent
        self.p_int_theta = np.tile(self.p_int, (config.K, 1))

        # self.E_obs = data_obs[:, 1] / (data_obs[:, 0] + data_obs[:, 1])     # E_obs(Y|X)
        self.E_exp = data_exp[:, 1] / (data_exp[:, 0] + data_exp[:, 1])     # E_exp(Y|do(X))
        self.experimental_successes = data_exp[:, 1]
        self.experimental_failures = data_exp[:, 0]
        self.N_data_exp = np.sum(data_exp, axis=1)  # Total experimental data for each action

        # ------------------------ #
        # Initialize data matrix   #
        # ------------------------ #
        # Exploration (4 arms x 4 Intents)
        self.s = np.ones((config.K, config.K))  # Successes
        self.f = np.ones((config.K, config.K))  # Failures

        if self.USE_OBSERVATIONAL_DATA is False:
            # Exploration only
            self.observed_successes = self.s
            self.observed_failures = self.f
        else:
            # Exploration + observation
            self.observed_successes = (self.s + np.diag(data_obs[:, 1])).astype(int)
            self.observed_failures = (self.f + np.diag(data_obs[:, 0])).astype(int)

        # Record results
        Intent = np.zeros(T)
        Action = np.zeros(T)  # Action (0-3) taken for each timestep
        Reward = np.zeros(T)  # Reward (0,1) for each timestep
        Prob = np.zeros(T)  # Best action or not (0,1) for each timestep
        AveragePayoutAccuracy = np.zeros(T)

        CumProb = np.zeros(T)

        self.N_data = (self.observed_successes + self.observed_failures).astype(int)  # Total KxK data count
        self.p_wins = self.observed_successes / self.N_data  # Proportion of success (visualisation purpose)

        # Counter for exploration (visualisation purpose)
        self.exploration_count = np.zeros((config.K, config.K))

        print('Loaded data. Activating agent... \n')

        # Execute exploration for each timestep t
        tt = 0
        for t in range(T):

            intent = I_samples[tt].astype(int)

            #----------------------------#
            # Update expected rewards P  #
            #----------------------------#
            # Choose action
            if t % config.N_BATCH == 0:
                # Update model and get new posterior
                # Choose action from posterior
                print("ALGORITHM:" + self.algorithm_name)
                print('Obtaining posterior, N = {}, t = {} \n'.format(self.n, t))
                y_post, model, trace = self.get_posterior()
                if config.ALL_TS:
                    sample = config.sigmoid(trace[-1]['c'])
                    choices = sample[:, intent]
                    action = np.argmax(choices)
                else:
                    choices = y_post[:, intent]
                    action = np.argmax(choices)

                # Record and print information
                self.exploration_count[action, intent] += 1
                print("N = {}, Exploration count = \n {} \n".format(self.n, self.exploration_count))
                print("N = {}, y_post = \n {} \n".format(self.n, y_post.round(3)))
                counter = 2  # For trace
                latest_updated_intent_index = tt
                tt += 1

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
                # choices = sample[:, a]
                # action = np.argmax(choices)

                # Or get sample from trace
                #print(trace[-counter]['c'])
                #sample = config.sigmoid(np.mean(trace[-counter]['c'], axis=0))
                # print(trace[-counter]['c'])

                intent = I_samples[latest_updated_intent_index].astype(int)  # Choose same intent as one used to update posterior

                if config.USE_MODEL_WITH_CONSTRAINT:
                    a_post = trace[-counter]['a'][:, intent]
                    b_post = trace[-counter]['b'][:, intent]
                    intercept_post = trace[-counter]['offset'][:, intent]
                    sample = config.sigmoid(a_post + b_post + intercept_post)
                    action = np.argmax(choices)
                else:
                    sample = config.sigmoid(trace[-counter]['c'])
                    choices = sample[:, intent]
                    action = np.argmax(choices)

                self.exploration_count[action, intent] += 1
                counter += 1

            # Probability of success
            win_prob = self.theta[action, intent]

            # Pull arm
            reward = np.random.choice(2, p=[1 - win_prob, win_prob])

            # # Update collected data matrix
            self.observed_successes[action, intent] += reward
            self.observed_failures[action, intent] += 1 - reward

            # Payout based on data only
            self.N_data = (self.observed_successes + self.observed_failures).astype(int)
            self.p_wins = self.observed_successes / self.N_data

            # Record
            Intent[t] = intent
            Action[t] = action
            Reward[t] = reward
            # [bestVal, bestAction] = max(theta(:, covariateIndex));
            bestAction = np.argmax(self.theta[:, intent])
            AveragePayoutAccuracy[t] = 1 - np.mean(np.abs(y_post - self.theta) / self.theta)

            if action == bestAction:
                Prob[t] = 1
                CumProb[t] = CumProb[t - 1] + 1
                print("BEST ACTION \n")
            else:
                Prob[t] = 0
                CumProb[t] = CumProb[t - 1]
                print("NOT BEST ACTION \n")

        CumProb = CumProb / np.arange(1, T + 1)

        return Intent, Action, Reward, Prob, CumProb, AveragePayoutAccuracy

    def get_posterior(self):

        if config.COMPARE_HYPERPARAMETERS is True:
            ALPHA_HYPER_GAMMA_SD = config.a[self.algorithm_index]
            BETA_HYPER_GAMMA_SD = config.b[self.algorithm_index]
        else:
            ALPHA_HYPER_GAMMA_SD = config.ALPHA_HYPER_GAMMA_SD
            BETA_HYPER_GAMMA_SD = config.BETA_HYPER_GAMMA_SD
        # ------------------------------------------------- #
        # USE PYMC3 TO CREATE POSTERIOR AND SAMPLE FROM IT  #
        # ------------------------------------------------- #
        print('Building model... \n')

        model = pm.Model()
        with model:

            if config.USE_MODEL_WITH_CONSTRAINT:
                # Priors for unknown model parameters
                hyper_mu = pm.Normal('hyper_mu', mu=0, sd=10)
                hyper_sd = pm.Gamma('hyper_sd', alpha=ALPHA_HYPER_GAMMA_SD,
                                    beta=BETA_HYPER_GAMMA_SD)
                a = pm.Normal('a', mu=hyper_mu, sd=hyper_sd, shape=(config.K, 1))
                b = pm.Normal('b', mu=hyper_mu, sd=hyper_sd, shape=(1, config.K))
                offset = pm.Normal('offset', mu=hyper_mu, sd=hyper_sd)

                c = a + b + offset
                p = config.sigmoid(c)

            else:
                # Priors for unknown model parameters
                hyper_mu = pm.Normal('hyper_mu', mu=0, sd=10)
                hyper_sd = pm.Gamma('hyper_sd', alpha=ALPHA_HYPER_GAMMA_SD,
                                    beta=BETA_HYPER_GAMMA_SD)
                c = pm.Normal('c', mu=hyper_mu, sd=hyper_sd, shape=(config.K, config.K))

                p = config.sigmoid(c)

            if self.USE_INTERVENTIONAL_DATA:
                # Observational and counterfactual likelihood
                pm.Binomial('L_obs_and_cf', n=self.N_data, p=p, observed=self.observed_successes)

                # Experimental P(I) prior
                # theta = pm.Dirichlet('theta', np.ones(config.K), shape=(1, 4))
                # theta = pm.Dirichlet('theta', data_exp[:, 0], shape=(1, config.K))
                thet = self.p_int_theta  # (1, 4)

                p_exp = pm.math.sum(p * thet, axis=1)

                # Experimental likelihood
                pm.Binomial('L_int', n=self.N_data_exp, p=p_exp, observed=self.experimental_successes)

            else:
                # Observational and counterfactual likelihood
                pm.Binomial('L_obs_and_cf', n=self.N_data, p=p, observed=self.observed_successes)

            print('Starting MCMC... \n')

            # Draw posterior samples
            if config.WINDOWS:
                trace = pm.sample(config.TRACE_LENGTH, nuts_kwargs=dict(target_accept=.9,
                                  max_treedepth=20), chains=config.N_MCMC_CHAINS, cores=1)
            else:
                trace = pm.sample(config.TRACE_LENGTH, nuts_kwargs=dict(target_accept=.9,
                                  max_treedepth=20), chains=config.N_MCMC_CHAINS)

        if config.USE_PPC_SAMPLES:
            # Use samples from ppc to obtain posterior point estimates
            ppc = pm.sample_ppc(trace, samples=config.N_PPC_SAMPLES, model=model)
            y_post = np.mean(ppc['L_obs_and_cf'], axis=0) / self.N_data

        else:
            # Use trace to obtain posterior point estimates (faster)
            c_ = np.array(np.mean(trace[:config.N_TRACE_SAMPLES]['c'], axis=0))
            y_post = config.sigmoid(c_)

        return y_post, model, trace

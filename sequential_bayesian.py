"""
MABUC (Bayesian) run file
"""

import numpy as np
import pymc3 as pm
# import theano.tensor as tt

from config import Config
config = Config()

# Load experimental and observational data
data_exp = config.data_exp
data_obs = config.data_obs

# Observational data
E_obs = config.E_obs     # E_obs(Y|X)
p_int = config.p_int     # P(I) from observational data, X=intent

# Experimental data
E_exp = config.E_exp     # E_exp(Y|do(X))
experimental_successes = config.experimental_successes
experimental_failures = config.experimental_failures
N_data_exp = config.N_data_exp  # Total experimental data for each action


class MabucBayesianSeqAlgorithm:
    '''
    Bayesian version of MABUC - MCMC sampling of posterior parameters using Hamiltonial Monte Carlo.
    Algorithms:
    * 'MCMC++' = use counterfactual, observational and experimental data
    * 'MCMC+'  = use counterfactual and observational data only
    * 'MCMC'   = use counterfactual data only
    '''

    def __init__(self, algorithm_name, algorithm_index):
        self.algorithm_name = algorithm_name
        self.algorithm_index = algorithm_index
        self.USE_INTERVENTIONAL_DATA = True if algorithm_name.startswith('MCMC++') else False
        self.USE_OBSERVATIONAL_DATA = True if algorithm_name.startswith('MCMC+') else False

        if 'seq' in algorithm_name:
            # Load sequential payout ground truth
            self.theta = config.SEQ_THETA
        else:
            # Load payout for vanilla Bayesian MABUC
            self.theta = config.THETA

    # Edited for sequential - t_sequence
    def run_simulation(self, n, T, t_sequence, I_samples):
        """
        Run one simulation for T timesteps (T exploration steps)
        """

        self.n = n
        THETA = config.SEQ_THETA[t_sequence, :, :]  # Edited for sequential
        #T = config.T * config.N_BATCH

        # # Instantiate (sample) confounders for all T steps
        # U_samples = np.random.randint(2, size=(config.N_CONFOUNDERS, config.T))
        # I_samples = np.array(config.intent(U_samples[0], U_samples[1]))

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
        Intent = np.zeros(T)        # Intent (0-3) seen for each timestep
        Action = np.zeros(T)        # Action (0-3) taken for each timestep
        Reward = np.zeros(T)        # Reward (0,1) observed for each timestep
        Conds = np.zeros(config.U)  # Confounder count

        Prob = np.zeros(T)     # Best action or not (0,1) for each timestep
        CumProb = np.zeros(T)  # Cumulative best action count

        self.N_data = (self.observed_successes + self.observed_failures).astype(int)  # Total KxK data count
        self.p_wins = self.observed_successes / self.N_data  # Proportion of success (visualisation purpose)
        # self.y_post = self.p_wins

        # Counter for exploration (visualisation purpose)
        self.exploration_count = np.zeros((config.K, config.K))

        # Save intent, action and reward for next sequence
        # IAYs = []

        # for t in range(config.T):
        t_posterior = 0
        for t in range(T):

            intent = I_samples[t_posterior].astype(int)
            # self.I = I_samples[t].astype(int)
            covariateIndex = intent
            Conds[covariateIndex] += 1

            #----------------------------#
            # Update expected rewards P  #
            #----------------------------#

            if t % config.N_BATCH == 0:
                # Update model and get new posterior
                # Choose action from posterior
                print('Obtaining posterior, N = {}, t = {} \n'.format(self.n + 1, t))
                self.y_post, model, trace = self.get_posterior()
                choices = self.y_post[:, intent]
                action = np.argmax(choices)

                # Record and print
                self.exploration_count[action, intent] += 1
                print("N = {}, Exploration count = \n {} \n".format(self.n + 1, self.exploration_count))
                print("N = {}, y_post = \n {} \n".format(self.n + 1, self.y_post.round(3)))
                trace_position = 1  # Positional counter for taking sample from trace for batch method (non-MC samples)
                t_posterior += 1

            else:

                intent = I_samples[t_posterior].astype(int)  # Choose same intent as one used to update posterior
                sample = config.sigmoid(trace[-trace_position]['c'])
                choices = sample[:, intent]
                action = np.argmax(choices)

                self.exploration_count[action, intent] += 1
                trace_position += 1

            # Probability of success
            win_prob = THETA[action, intent]

            # Pull arm
            reward = np.random.choice(2, p=[1 - win_prob, win_prob])

            # Save I, A and Y for next sequence
            # IAYs.append(intent)
            # IAYs.append(action)
            # IAYs.append(reward)

            # Update observed data
            self.observed_successes[action, intent] += reward
            self.observed_failures[action, intent] += 1 - reward

            # Update payout based on data only
            self.N_data[action, intent] += 1
            self.p_wins = self.observed_successes / self.N_data

            # Record results
            Intent[t] = intent
            Action[t] = action
            Reward[t] = reward

            best_action = np.argmax(THETA[:, intent])
            if action == best_action:
                Prob[t] = 1
                CumProb[t] = CumProb[t - 1] + 1
                print("BEST ACTION \n")
            else:
                Prob[t] = 0
                CumProb[t] = CumProb[t - 1]
                print("NOT BEST ACTION \n")

            CumProb = CumProb / np.arange(1, T + 1)

        # Prepare collected IAYs
        # IAYs = np.array(IAYs)
        # IAYs.reshape((-1, 3))

        return Intent, Action, Reward, Prob, CumProb, Conds

    def get_next_intents(self, Intent, Action, Reward):
        """
        Takes I, A and R (each length T) and returns intent for next sequential step
        """

        # Stack intents, actions and rewards seen in previous step
        IAYs = np.stack((Intent, Action, Reward)).T  # (T, 3)

        # Find indices in permutation list for each (I, A, Y)
        permutation_indices = np.where((config.IAY_permutations == IAYs[:, None]).all(-1))[1]

        # Get class probabilities list for each permutation
        class_probabilities = config.next_intent_probs

        # Get list of next intents
        next_intent_list = np.squeeze([np.random.choice(config.K, 1, p=class_probabilities[i, :]).astype(int)
                                       for i in permutation_indices])

        return next_intent_list

    # ------------------------------------------------- #
    #  Perform MCMC                                     #
    # ------------------------------------------------- #
    def get_posterior(self):
        """
        Performs MC sampling to return posterior, model and trace
        """

        if config.COMPARE_HYPERPARAMETERS is True:
            ALPHA_HYPER_GAMMA_SD = config.a[self.algorithm_index]
            BETA_HYPER_GAMMA_SD = config.b[self.algorithm_index]
        else:
            ALPHA_HYPER_GAMMA_SD = config.ALPHA_HYPER_GAMMA_SD
            BETA_HYPER_GAMMA_SD = config.BETA_HYPER_GAMMA_SD

        # ------------------------------------------------- #
        # USE PYMC3 TO CREATE POSTERIOR AND SAMPLE FROM IT  #
        # ------------------------------------------------- #
        model = pm.Model()
        with model:

            # Set priors for unknown model parameters
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
                theta = pm.Dirichlet('theta', data_exp[:, 0], shape=(1, 4))

                p_exp = pm.math.sum(p * theta, axis=1)

                # Experimental likelihood
                pm.Binomial('L_int', n=N_data_exp, p=p_exp, observed=experimental_successes)

            else:
                # Observational and counterfactual likelihood
                pm.Binomial('L_obs_and_cf', n=self.N_data, p=p, observed=self.observed_successes)

            # Draw posterior samples
            trace = pm.sample(config.TRACE_LENGTH, nuts_kwargs=dict(target_accept=.9,
                              max_treedepth=20), chains=config.N_MCMC_CHAINS)

        # Use trace to obtain posterior point estimates
        c_ = np.array(np.mean(trace[:config.N_TRACE_SAMPLES]['c'], axis=0))
        y_post = config.sigmoid(c_)

        return y_post, model, trace

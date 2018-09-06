import numpy as np
# from datasim import create_data


class Config:
    '''
    Configuration file for simulation
    '''

    def __init__(self):

        self.MODE = 'vanilla'         # Simulate with specified payout rates
        #self.MODE = 'ihdp'              # Simulate with IHDP data

        # If vanilla:
        self.USE_RANDOM_DATA = True        # If true, creates obs/exp data and theta for EACH N
        self.USE_SAVED_RANDOM_DATA = True  # Use already saved data
        self.SAVE_LIST = False             # Save random payout and data generated

        self.WINDOWS = False             # Set to True if using Windows OS

        self.SAVE_CHECKPOINTS = True    # IMPORTANT! Set to true if you want to save data as files
        self.SAVE_AT_END = False        # May want to set to true if doing IVWA

        self.SAVE_DATA = True           # Old

        self.PLOT = False               # Plot graph at the end of simulations

        # -----------------------------------------------#
        #  Create observational and experimental data?   #
        # -----------------------------------------------#
        # Only relevant for 'vanilla'
        # self.USE_RANDOM_DATA = True        # If true, creates obs/exp data and theta for EACH N
        # self.SAVE_LIST = False             # Save random payout and data generated
        # self.USE_SAVED_RANDOM_DATA = True  # Use already saved data

        self.SIMULATE_OBS_EXP_DATA = False  # If true, creates and saves obs/exp data
        self.SAMPLES_PER_ARM = 10000        # Samples per arm for data generation
        self.CREATE_OBS_EXP_DATA_ONLY = False  # If true, comparison algorithm will also run

        # -----------------------------------------------#
        #  MABUC run main settings                       #
        # -----------------------------------------------#
        self.T = 150      # Exploration timesteps
        self.N = 20      # MC repeats to average (if using existing I samples and this
                        # includes multiple N, this will be multiplied by this)
        self.USE_ORDERED_I_SAMPLES = True  # If false, I appears in order

        # Available:
        #self.ALGORITHMS = ['IVWA - TS++', 'IVWA - TS+', 'IVWA - TS',
        #                   'IVWA - paper++', 'IVWA - paper+', 'IVWA - paper']
        # self.ALGORITHMS = ['MCMC++']
        self.ALGORITHMS = ['MCMC++']

        # OR

        # Compare Bayesian hyperparameters (set to false if using IVWA)
        # If this is set true, each combination of (a, b) is treated as
        # separate algorithms with base algorithm base_alg.
        # self.ALGORITHMS set in this config file is overwritten.
        self.COMPARE_HYPERPARAMETERS = False
        self.a = [10]
        self.b = [1]
        self.base_alg = 'MCMC++'

        # Use existing intent instantiations
        # (this is autoamtically true if using "compare hyperparameters")
        self.USE_EXISTING_I_SAMPLES = False
        self.I_SAMPLE_FILE = 'I_samples_TEST2.npy'

        # Save intent instantiations if not using existing samples?
        self.SAVE_I_SAMPLES = False

        # -------------------------------- #
        #  Basic simulation settings       #
        # -------------------------------- #

        # self.N_MCMC = 3  # Number of rounds for MCMC method
        # self.N_IVWA = 3  # Number of rounds of inverse variance weighted average method

        self.K = 4                                  # Number of arms
        self.N_CONFOUNDERS = int(np.log2(self.K))   # Number of unobserved confounders. Originally U, 2 for D,B
        self.U = 2 ** self.N_CONFOUNDERS            # Number of unique confounder instantiations. Originally uConds,
        self.N_OBS = self.K * self.SAMPLES_PER_ARM  # Number of observational samples, per arm
        self.N_EXP = self.K * self.SAMPLES_PER_ARM  # Number of experimental samples, per arm

        # Payout setting for non-sequential runs
        if self.MODE == 'vanilla':
            self.THETA = np.array([[0.2, 0.3, 0.5, 0.6],
                                   [0.6, 0.2, 0.3, 0.5],
                                   [0.5, 0.6, 0.2, 0.3],
                                   [0.3, 0.5, 0.6, 0.2]])

        if self.MODE == 'ihdp':
            self.THETA = np.load('ihdp_payout.npy')         # Payout ground truth

        # self.THETA = np.array([[ 0.2,  0.3,  0.5,  0.3,  0.2,  0.6],
        #                        [ 0.6,  0.2,  0.3,  0.5,  0.3,  0.2],
        #                        [ 0.5,  0.6,  0.2,  0.3,  0.5,  0.3],
        #                        [ 0.3,  0.5,  0.6,  0.2,  0.3,  0.5],
        #                        [ 0.2,  0.3,  0.5,  0.6,  0.2,  0.3],
        #                        [ 0.3,  0.2,  0.3,  0.5,  0.6,  0.2]])

        # self.THETA = np.array([[ 0.3,  0.3,  0.5,  0.6,  0.2,  0.4],
        #                        [ 0.2,  0.6,  0.3,  0.4,  0.4,  0.6],
        #                        [ 0.4,  0.4,  0.3,  0.2,  0.3,  0.5],
        #                        [ 0.6,  0.3,  0.4,  0.3,  0.6,  0.3],
        #                        [ 0.3,  0.2,  0.6,  0.3,  0.3,  0.3],
        #                        [ 0.5,  0.5,  0.2,  0.5,  0.5,  0.2]])

        #--------------------------------------------#
        # Load experimental and observational data   #
        #--------------------------------------------#

        if self.MODE == 'vanilla':

            if self.USE_RANDOM_DATA:
                # Data is generated in the main file
                self.data_exp = None
                self.data_obs = None
            else:
                # Load observational and experimental data
                if self.K == 6:
                    self.data_exp = np.load('data_exp_6.npy')
                    self.data_obs = np.load('data_obs_6.npy')
                elif self.K == 4:
                    self.data_exp = np.load('data_exp.npy')
                    self.data_obs = np.load('data_obs.npy')
                else:
                    print("K must be 4 or 6")

        if self.MODE == 'ihdp':
            self.data_exp = np.load('data_exp_ihdp.npy')    # Experimental data
            self.data_obs = np.load('data_obs_ihdp.npy')    # Observational data

            # # Observational data
            # self.E_obs = self.data_obs[:, 1] / (self.data_obs[:, 0] + self.data_obs[:, 1])     # E_obs(Y|X)
            # self.p_int = np.sum(self.data_obs, axis=1) / np.sum(self.data_obs)  # P(I) from observational data, X=intent
            #
            # # Experimental data
            # self.E_exp = self.data_exp[:, 1] / (self.data_exp[:, 0] + self.data_exp[:, 1])     # E_exp(Y|do(X))
            # self.experimental_successes = self.data_exp[:, 1]
            # self.experimental_failures = self.data_exp[:, 0]
            # self.N_data_exp = np.sum(self.data_exp, axis=1)  # Total experimental data for each action

        # else:  # If payout is different for each N
        #     self.data_exp = None  #
        #     self.data_obs = None  #
        #     self.E_obs = None  #
        #     self.p_int = None  #
        #     self.E_exp = 0  #
        #     self.experimental_successes = None  #
        #     self.experimental_failures = None  #
        #     self.N_data_exp = None  #

        # -------------------------------------------- #
        #  PYMC3 BAYESIAN MABUC SETTINGS               #
        # -------------------------------------------- #
        self.USE_MODEL_WITH_CONSTRAINT = False  # Use alpha + beta + theta model
        self.N_BATCH = 2                    # Batch size for taking data from trace with each MC update of posterior
        self.TRACE_LENGTH = 600             # Trace length for MCMC
        self.N_MCMC_CHAINS = 2              # Chain length for MCMC
        self.USE_PPC_SAMPLES = False        # (UNUSED) Use PPC for posterior point estimate
        self.N_PPC_SAMPLES = 500            # (UNUSED) How many PPC samples for posterior
        self.N_TRACE_SAMPLES = 100          # How many of last trace samples to use to get posterior
        self.ALPHA_HYPER_GAMMA_SD = 100     # Alpha in hyperprior sd gamma(alpha, beta)
        self.BETA_HYPER_GAMMA_SD = 10       # Beta in hyperprior sd gamma(alpha, beta)
        self.ALL_TS = True                  # Thompson sample for every t
        # Been using 0.1, 0.01 / 0.01, 0.001 / 100, 10

        # (10, 0.1), (10, 0.001) DOESN"T EXPLORE ENOUGH AT THE BEGINNING!
        # LESS INFO PRIOR NEEDED

        # -------------------------------------------- #
        # SEQUENTIAL BAYESIAN SETTINGS                 #
        # -------------------------------------------- #

        # Payouts for sequential MABUC
        self.SEQ_LENGTH = 2
        self.theta1_ = np.array([[0.2, 0.3, 0.5, 0.6],
                                 [0.6, 0.2, 0.3, 0.5],
                                 [0.5, 0.6, 0.2, 0.3],
                                 [0.3, 0.5, 0.6, 0.2]])

        self.theta2_ = np.array([[0.2, 0.3, 0.5, 0.6],
                                 [0.6, 0.2, 0.3, 0.5],
                                 [0.5, 0.6, 0.2, 0.3],
                                 [0.3, 0.5, 0.6, 0.2]])

        self.SEQ_THETA = np.stack((self.theta1_, self.theta2_))  # (2, 4, 4)

        # All permutations of intents, actions and rewards (used in sequential method)
        intents = np.arange(self.K)
        actions = np.arange(self.K)
        rewards = np.arange(2)
        self.IAY_permutations = np.array(np.meshgrid(intents, actions, rewards)).T.reshape(-1, 3)

        # Random probabilities for next intent
        np.random.seed(100)
        self.next_intent_probs = np.random.dirichlet(np.ones(self.K), size=self.IAY_permutations.shape[0])

        # ======================== DO NOT EDIT BELOW =========================

        # If comparing hyperparameters, overwrite self.ALGORITHMS
        if self.COMPARE_HYPERPARAMETERS:
            self.ALGORITHMS = []
            for i in range(len(self.a)):
                self.ALGORITHMS.append(self.base_alg + ', a={}, b={}'.format(self.a[i], self.b[i]))

        # Number of algorithms to run
        self.N_ALGS = len(self.ALGORITHMS)

        # Set timesteps depending on MCMC or IWVA
        # For MCMC, we also run additional timesteps (N_BATCH) for each MC as a 'batch'
        # and the timesteps required for this is equal to the length of I_samples_list
        # variable fed into the algorithm.
        if self.ALGORITHMS[0].startswith('MCMC') and self.USE_EXISTING_I_SAMPLES:
            self.T_ = 'I_samples_list.shape[1]'
        elif self.ALGORITHMS[0].startswith('MCMC') and not self.USE_EXISTING_I_SAMPLES:
            self.T_ = 'config.T * config.N_BATCH'
        else:
            self.T_ = 'config.T'

        # Do not use random data if we are using ihdp data
        if self.MODE == 'ihdp':
            self.USE_RANDOM_DATA = False

    # ---------------------------------- #
    #  Other methods                     #
    # ---------------------------------- #

    @staticmethod
    def intent(D, B):
        """ Confounders to intent mechanism """
        return D + 2 * B

    @staticmethod
    def covariate_index(D, B):  # Not used
        """ Confounders to intent mechanism """
        return D + 2 * B

    @staticmethod
    def sigmoid(x):
        """ Numerically stable sigmoid """
        z = np.exp(x)
        return z / (1 + z)

    @staticmethod
    def crazyshuffle(arr):
        """ Shuffle each column separately """
        x, y = arr.shape
        rows = np.indices((x,y))[0]
        cols = [np.random.permutation(y) for _ in range(x)]
        return arr[rows, cols].T

"""

Main MABUC experiment run file (python 3.6)
To run experiments do
> python main.py
PLEASE SET YOUR CONFIGURATIONS IN config.py
GENERATE YOUR OBS/EXP DATA FIRST in datasim.py

"""

import numpy as np
# import pymc3 as pm
# from numpy.linalg import inv
# from scipy.stats import beta
import matplotlib.pyplot as plt
# import math
# from collections import defaultdict
# import sys
import datetime
# import time
import os

# My own modules:
from datasim import generate_and_save_n_data
# from datasim import generate_and_save_data
from config import Config
from helper import smooth
# from helper import tic, toc
# import mabuc
import mabuc_clean
# import mabuc_bayesian
import mabuc_bayesian_clean
# import sequential_bayesian


def main():

    # Load settings
    config = Config()

    # Get date-time stamp
    stamp = str(datetime.datetime.now()).split('.')[0]
    stamp = stamp.replace(':','-')
    stamp = stamp.replace(' ','-')

    # Data save folder
    save_path = 'data_folder'

    # ---------------------------------------------------------------------- #
    # Compare algorithms - Instantiate intent for all timesteps              #
    # ---------------------------------------------------------------------- #
    if config.USE_EXISTING_I_SAMPLES:  # Load I samples from existing file
        I_samples_list = np.load(os.path.join(save_path, config.I_SAMPLE_FILE))[:, :config.T * config.N_BATCH]
        I_samples_list = np.tile(I_samples_list, (config.N, 1))  # Tile N times
        T = eval(config.T_)  # This needs I_samples_list
        N = config.N
    else:  # Instantiate confounders and I samples to use for all timesteps
        T = eval(config.T_)
        U_samples_list = np.empty((config.N, config.N_CONFOUNDERS, T))
        I_samples_list = np.empty((config.N, T))
        N = config.N

        for i in range(config.N):
            U_samples = np.random.randint(2, size=(config.N_CONFOUNDERS, T))
            I_samples = np.array(config.intent(U_samples[0], U_samples[1]))
            U_samples_list[i] = U_samples
            I_samples_list[i] = I_samples

        if config.USE_ORDERED_I_SAMPLES:
            # Systematic pattern of intents
            I_patterns = np.zeros((config.K, config.K)).astype(int)
            for i in range(config.K):
                I_patterns[i] = np.roll(np.arange(config.K, dtype=np.int), -i)
            I_patterns = np.tile(I_patterns, np.ceil(T / config.K).astype(int))
            I_patterns = np.tile(I_patterns, (np.ceil(N / config.K).astype(int), 1))
            I_samples_list = I_patterns[:N, :T]

        if config.SAVE_I_SAMPLES:
            np.save(os.path.join(save_path, 'I_samples_' + stamp), I_samples_list)

    # ------------------------------------------------------ #
    #  Prepare run for T timesteps N times                   #
    # ------------------------------------------------------ #

    if config.COMPARE_HYPERPARAMETERS:
        N = config.N
    else:
        # If config.USE_RANDOM_I_SAMPLES, will NOT WORK
        N = I_samples_list.shape[0] if config.USE_EXISTING_I_SAMPLES is True else config.N

    if config.MODE == 'vanilla' and config.USE_RANDOM_DATA:
        if config.USE_SAVED_RANDOM_DATA:
            data_exp_list = np.load('data_exp_list.npy')
            data_obs_list = np.load('data_obs_list.npy')
            theta_list = np.load('theta_list.npy')
            N = len(theta_list)

        else:
            data_exp_list, data_obs_list, theta_list = generate_and_save_n_data(config.N, save=config.SAVE_LIST, noisy=False)

        data_lists = [data_exp_list, data_obs_list, theta_list]
    else:
        data_lists = None

    # Sums of actions, rewards, and times best action chosen.
    IntentSum = np.zeros((config.N_ALGS, T))
    ActionSum = np.zeros((config.N_ALGS, T))
    RewardSum = np.zeros((config.N_ALGS, T))
    ProbSum = np.zeros((config.N_ALGS, T))
    RegretSum = np.zeros((config.N_ALGS, T))
    CumRegretSum = np.zeros((config.N_ALGS, T))
    CumProbSum = np.zeros((config.N_ALGS, T))
    AccuracySum = np.zeros((config.N_ALGS, T))
    ExpectedRewardSum = np.zeros((config.N_ALGS, T))

    if config.PLOT:
        plt.figure()

    # -------------------------------------------------- #
    #  Run for T timesteps N times for all algorithms    #
    # -------------------------------------------------- #

    for a in range(config.N_ALGS):

        alg_name = config.ALGORITHMS[a]
        print('Running algorithm: {} \n'.format(alg_name))

        # -------------------------------- #
        #  Instantiate algorithm           #
        # -------------------------------- #

        if alg_name.startswith('MCMC'):
            # algorithm = mabuc_bayesian.MabucBayesianAlgorithm(alg_name, a)  # <-- doesn't work with random data
            algorithm = mabuc_bayesian_clean.MabucBayesianAlgorithm(alg_name, a)
            T = config.T * config.N_BATCH
            # N = config.N_MCMC

        else:
            # algorithm = mabuc.MabucAlgorithm(alg_name)
            algorithm = mabuc_clean.MabucAlgorithm(alg_name)
            T = config.T
            # N = config.N_IVWA

        # For calculating sample standard deviation

        Prob_log = np.zeros((N, T))
        Regret_log = np.zeros((N, T))
        CumRegret_log = np.zeros((N, T))
        ExpectedReward_log = np.zeros((N, T))
        SmoothedCumRegret_log = np.zeros((N, T))
        Accuracy_log = np.zeros((N, T))

        # -------------------------------- #
        # Run N Monte Carlo Simulations    #
        # -------------------------------- #

        print('Executing {} MC Simulations... \n'.format(N))

        for n in range(N):

            print('======  N = {} ====== \n'.format(n))

            # Run T exploration steps
            Intent, Action, Reward, Prob, CumProb, AveragePayoutAccuracy = algorithm.run_simulation(n, T, I_samples, data_lists)

            # Collect stats
            IntentSum[a, :] += Intent
            ActionSum[a, :] += Action
            RewardSum[a, :] += Reward

            ProbSum[a, :] += Prob
            CumProbSum[a, :] += CumProb
            AccuracySum[a, :] += AveragePayoutAccuracy

            # Save regret for this simulation (N)
            optimal_rewards = np.max(config.THETA[:, Intent.astype(int)], axis=0)
            action_rewards = config.THETA[Action.astype(int), Intent.astype(int)]
            Regret = optimal_rewards - action_rewards
            RegretSum[a, :] += Regret
            CumRegret = np.cumsum(optimal_rewards - action_rewards)
            CumRegretSum[a, :] += CumRegret
            ExpectedRewardSum[a, :] += action_rewards

            Prob_log[n, :] = Prob
            Regret_log[n, :] = Regret
            CumRegret_log[n, :] = CumRegret
            ExpectedReward_log[n, :] = action_rewards
            Accuracy_log[n, :] = AveragePayoutAccuracy

            if config.SAVE_CHECKPOINTS:

                print('Saving checkpoint... \n')

                # np.save(os.path.join(save_path, 'checkpoint', 'N = ' + str(n) + '_PProb_' + config.ALGORITHMS[a] + '_' + stamp), Prob)
                # np.save(os.path.join(save_path, 'checkpoint', 'N = ' + str(n) + '_Regret_' + config.ALGORITHMS[a] + '_' + stamp), Regret)
                # np.save(os.path.join(save_path, 'checkpoint', 'N = ' + str(n) + '_CumRegret_' + config.ALGORITHMS[a] + '_' + stamp), CumRegret)
                # np.save(os.path.join(save_path, 'checkpoint', 'N = ' + str(n) + '_Accuracy_' + config.ALGORITHMS[a] + '_' + stamp), AveragePayoutAccuracy)
                #
                # np.save(os.path.join(save_path, 'checkpoint', 'AVE_N = ' + str(n) + '_PProb_' + config.ALGORITHMS[a] + '_' + stamp), ProbSum[a, :] / (n + 1))
                # np.save(os.path.join(save_path, 'checkpoint', 'AVE_N = ' + str(n) + '_Regret_' + config.ALGORITHMS[a] + '_' + stamp), RegretSum[a, :] / (n + 1))
                # np.save(os.path.join(save_path, 'checkpoint', 'AVE_N = ' + str(n) + '_CumRegret_' + config.ALGORITHMS[a] + '_' + stamp), CumRegretSum[a, :] / (n + 1))
                # np.save(os.path.join(save_path, 'checkpoint', 'AVE_N = ' + str(n) + '_Accuracy_' + config.ALGORITHMS[a] + '_' + stamp), AccuracySum[a, :] / (n + 1))

                folder = './' + save_path + '/' + config.MODE + '/'
                np.save(folder + stamp + '_' + config.ALGORITHMS[a] + '_AVE_N=' + str(n) + '_PProb' , ProbSum[a, :] / (n + 1))
                np.save(folder + stamp + '_' + config.ALGORITHMS[a] + '_AVE_N=' + str(n) + '_Regret', RegretSum[a, :] / (n + 1))
                np.save(folder + stamp + '_' + config.ALGORITHMS[a] + '_AVE_N=' + str(n) + '_CumRegret', CumRegretSum[a, :] / (n + 1))
                np.save(folder + stamp + '_' + config.ALGORITHMS[a] + '_AVE_N=' + str(n) + '_Accuracy', AccuracySum[a, :] / (n + 1))

                if n >= 1:
                    STD_PProb = np.std(Prob_log[:(n + 1)], axis=0, ddof=1)  # Sample std
                    STD_Regret = np.std(Regret_log[:(n + 1)], axis=0, ddof=1)  # Sample std
                    STD_CumRegret = np.std(CumRegret_log[:(n + 1)], axis=0, ddof=1)  # Sample std
                    STD_Accuracy = np.std(Accuracy_log[:(n + 1)], axis=0, ddof=1)  # Sample std

                    np.save(folder + stamp + '_' + config.ALGORITHMS[a] + '_STD_N=' + str(n) + '_PProb' , STD_PProb)
                    np.save(folder + stamp + '_' + config.ALGORITHMS[a] + '_STD_N=' + str(n) + '_Regret', STD_Regret)
                    np.save(folder + stamp + '_' + config.ALGORITHMS[a] + '_STD_N=' + str(n) + '_CumRegret', STD_CumRegret)
                    np.save(folder + stamp + '_' + config.ALGORITHMS[a] + '_STD_N=' + str(n) + '_Accuracy', STD_Accuracy)

        # --------------------------------------- #
        #  Save results to plot                   #
        # --------------------------------------- #

        if config.SAVE_ALL_AT_END:

            # Get Index of best and worst runs
            for n in range(N):
                SmoothedCumRegret_log[n] = smooth(CumRegret_log[n])
            BestRun_idx = np.argmin(SmoothedCumRegret_log[:, -1])
            WorstRun_idx = np.argmax(SmoothedCumRegret_log[:, -1])

            #  Probability of optimal action
            # -----------------------------------------------
            PProb = ProbSum[a, :] / N
            STD_PProb = np.std(Prob_log, axis=0, ddof=1)  # Sample std
            name = config.ALGORITHMS[a] + '_' + stamp + '_PProb'
            name_std = config.ALGORITHMS[a] + '_' + stamp + '_PProb_std'
            if config.SAVE_DATA:
                np.save(os.path.join(save_path, name), PProb)
                np.save(os.path.join(save_path, name_std), STD_PProb)

            #  Save cumulative probability of optimal action
            # -----------------------------------------------
            CumPProb = CumProbSum[a, :] / N
            name = config.ALGORITHMS[a] + '_' + stamp + '_CumPProb'
            if config.SAVE_DATA:
                np.save(os.path.join(save_path, name), CumPProb)

            #  Save regret
            # -----------------------------------------------
            Regret = RegretSum[a, :] / N
            STD_Regret = np.std(Regret_log, axis=0, ddof=1)  # Sample std

            Regret_Best = Regret_log[BestRun_idx]
            Regret_Worst = Regret_log[WorstRun_idx]

            if config.SAVE_DATA:
                name = config.ALGORITHMS[a] + '_' + stamp + '_Regret'
                name_std = config.ALGORITHMS[a] + '_' + stamp + '_Regret_std'
                np.save(os.path.join(save_path, name), Regret)
                np.save(os.path.join(save_path, name_std), STD_Regret)

                name_best = config.ALGORITHMS[a] + '_' + stamp + '_Regret_Best'
                name_worst = config.ALGORITHMS[a] + '_' + stamp + '_Regret_Worst'
                np.save(os.path.join(save_path, name_best), Regret_Best)
                np.save(os.path.join(save_path, name_worst), Regret_Worst)

            #  Cumulative regret
            # -----------------------------------------------
            CumRegret = CumRegretSum[a, :] / N
            STD_CumRegret = np.std(CumRegret_log, axis=0, ddof=1)  # Sample std

            CumRegret_Best = CumRegret_log[BestRun_idx]
            CumRegret_Worst = CumRegret_log[WorstRun_idx]

            if config.SAVE_DATA:
                name = config.ALGORITHMS[a] + '_' + stamp + '_CumRegret'
                name_std = config.ALGORITHMS[a] + '_' + stamp + '_CumRegret_std'
                np.save(os.path.join(save_path, name), CumRegret)
                np.save(os.path.join(save_path, name_std), STD_CumRegret)

                name_best = config.ALGORITHMS[a] + '_' + stamp + '_CumRegret_Best'
                name_worst = config.ALGORITHMS[a] + '_' + stamp + '_CumRegret_Worst'
                np.save(os.path.join(save_path, name_best), CumRegret_Best)
                np.save(os.path.join(save_path, name_worst), CumRegret_Worst)

            #  Payout estimate accuracy
            # -----------------------------------------------
            Accuracy = AccuracySum[a, :] / N
            STD_Accuracy = np.std(Accuracy_log, axis=0, ddof=1)  # Sample std
            name = config.ALGORITHMS[a] + '_' + stamp + '_Accuracy'
            name_std = config.ALGORITHMS[a] + '_' + stamp + '_Accuracyt_std'
            if config.SAVE_DATA:
                np.save(os.path.join(save_path, name), Accuracy)
                np.save(os.path.join(save_path, name_std), STD_Accuracy)

            #  Action rewards
            # -----------------------------------------------
            ExpectedReward = ExpectedRewardSum[a, :] / N
            STD_ExpectedReward = np.std(ExpectedReward_log, axis=0, ddof=1)  # Sample std

            ExpectedReward_Best = ExpectedReward_log[BestRun_idx]
            ExpectedReward_Worst = ExpectedReward_log[WorstRun_idx]

            if config.SAVE_DATA:
                name = config.ALGORITHMS[a] + '_' + stamp + '_ExpectedReward'
                name_std = config.ALGORITHMS[a] + '_' + stamp + '_ExpectedReward_std'
                np.save(os.path.join(save_path, name), ExpectedReward)
                np.save(os.path.join(save_path, name_std), STD_ExpectedReward)

                name_best = config.ALGORITHMS[a] + '_' + stamp + '_ExpectedReward_Best'
                name_worst = config.ALGORITHMS[a] + '_' + stamp + '_ExpectedReward_Worst'
                np.save(os.path.join(save_path, name_best), ExpectedReward_Best)
                np.save(os.path.join(save_path, name_worst), ExpectedReward_Worst)

        # ---------------------------------- #
        #  Plot graphs                       #
        # ---------------------------------- #

        if config.PLOT:

            ax1 = plt.subplot(321)
            ax1.plot(smooth(PProb), label=config.ALGORITHMS[a])
            # ax1.set_yticks(np.arange(0, 1., 0.1))
            # ax1.set_xticks(np.arange(0, T, 50))
            ax1.grid()

            ax2 = plt.subplot(322)
            ax2.plot(smooth(CumPProb), label=config.ALGORITHMS[a])
            #ax2.set_yticks(np.arange(0, 100., 20))
            #ax2.set_xticks(np.arange(0, T, 50))
            ax2.grid()

            ax3 = plt.subplot(323)
            ax3.plot(smooth(Regret), label=config.ALGORITHMS[a])
            #ax3.set_yticks(np.arange(0, 100., 20))
            #ax3.set_xticks(np.arange(0, T, 50))
            ax3.grid()

            ax4 = plt.subplot(324)
            ax4.plot(smooth(CumRegret), label=config.ALGORITHMS[a])
            #ax4.set_yticks(np.arange(0, 100., 20))
            #ax4.set_xticks(np.arange(0, T, 50))
            ax4.grid()

            ax5 = plt.subplot(325)
            ax5.plot(smooth(Accuracy), label=config.ALGORITHMS[a])
            #ax5.set_yticks(np.arange(0, 100., 20))
            #ax5.set_xticks(np.arange(0, T, 50))
            ax5.grid()

            ax6 = plt.subplot(326)
            ax6.plot(smooth(ExpectedReward), label=config.ALGORITHMS[a])
            #ax6.set_yticks(np.arange(0, 100., 20))
            #ax6.set_xticks(np.arange(0, T, 50))
            ax6.grid()

            plt.subplot(326)
            #plt.fill_between(np.arange(T), smooth(ExpectedReward) - smooth(STD_ExpectedReward), smooth(ExpectedReward) + smooth(STD_ExpectedReward), color='b', alpha=0.2)
            plt.fill_between(np.arange(T), smooth(ExpectedReward_Best), smooth(ExpectedReward_Worst), color='b', alpha=0.2)

    if config.PLOT:
        plt.subplot(321)
        plt.xlabel('t')
        plt.ylabel('Probability of optimal action')
        plt.xlim(0, T)
        plt.legend()

        plt.subplot(322)
        plt.xlabel('t')
        plt.ylabel('Cumulative Probability')
        plt.xlim(0, T)
        plt.legend()

        plt.subplot(323)
        plt.xlabel('t')
        plt.ylabel('Current Regret')
        plt.xlim(0, T)
        plt.legend()

        plt.subplot(324)
        plt.xlabel('t')
        plt.ylabel('Total regret')
        plt.xlim(0, T)
        plt.legend()

        plt.subplot(325)
        plt.xlabel('t')
        plt.ylabel('Accuracy of Payout')
        plt.xlim(0, T)
        plt.legend()

        plt.subplot(326)
        plt.xlabel('t')
        plt.ylabel('Expected Reward')
        plt.xlim(0, T)
        plt.legend()

        plt.show()


if __name__ == '__main__':
    main()

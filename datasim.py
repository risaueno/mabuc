from config import Config
import numpy as np
# from get_intent import intent
# from get_intent import covariate_index

config = Config()

# data_exp_list, data_obs_list, theta_list = create_shuffled_data()
# np.save('data_exp_list', data_exp_list)
# np.save('data_obs_list', data_obs_list)
# np.save('data_exp_list', theta_list)

# data_exp, data_obs = create_data()
# np.save('data_exp_6', data_exp)
# np.save('data_obs_6', data_obs)

# generate_and_save_data()
# generate_and_save_n_data()

#data_exp, data_obs = create_data()
#np.save('data_exp', data_exp)
#np.save('data_obs', data_obs)


def generate_and_save_data(save=True):
    #if config.SIMULATE_OBS_EXP_DATA:
    data_exp, data_obs = create_data()
    if save:
        np.save('data_exp', data_exp)
        np.save('data_obs', data_obs)
    # if config.CREATE_OBS_EXP_DATA_ONLY:
    #     sys.exit()

    return data_exp, data_obs


def generate_and_save_n_data(N, save=True, noisy=False):
    """
    Create shuffled payout and data for each N
    """
    #if config.USE_RANDOM_DATA:
    data_exp_list, data_obs_list, theta_list = create_shuffled_data(N, noisy)
    if save:
        np.save('data_exp_list', data_exp_list)
        np.save('data_obs_list', data_obs_list)
        np.save('theta_list', theta_list)

    return data_exp_list, data_obs_list, theta_list


def create_data(theta=config.THETA, noisy=False):
    """
    Generate samples for observational and experimental data
    """

    data_exp = sample('experimental', theta, noisy)
    data_obs = sample('observational', theta, noisy)

    return data_exp, data_obs


def create_shuffled_data(N, noisy=False):
    # Will not work if use existing I_samples because (in main):
    # N = I_samples_list.shape[0] if config.USE_EXISTING_I_SAMPLES is True
    data_exp_list = []
    data_obs_list = []
    theta_list = []
    if config.K == 6:
        payouts = np.tile(np.array([0.2, 0.3, 0.3, 0.5, 0.6, 0.4]), (6, 1))
    elif config.K == 4:
        payouts = np.tile(np.array([0.2, 0.3, 0.6, 0.5]), (4, 1))
    else:
        print("K must be 4 or 6")

    for _ in range(N):
        shuffled_payout = config.crazyshuffle(payouts)
        theta_list.append(shuffled_payout)
        exp, obs = create_data(theta=shuffled_payout, noisy=noisy)
        data_exp_list.append(exp)
        data_obs_list.append(obs)

    return data_exp_list, data_obs_list, theta_list


def sample(data_type, theta, noisy):
    """
    noisy: samples, rather than computes obs E(Y|X) and exp E(Y|do(X))
    """

    if data_type not in ['observational', 'experimental']:
        raise ValueError("Data type must be 'observational' or 'experimental'")

    samples = np.zeros((config.K, 2))  # Columns for Y = 0 or 1

    if noisy:
        # (1) Confounder combinations are sampled with equal probability
        # (2) Reward is given with probability of winning according to payout specification

        # Sample confounder combinations with equal probability for D and B
        #U_samples = np.random.randint(
        #    2, size=(config.N_CONFOUNDERS, config.N_EXP))

        for i in range(config.N_EXP):
            # Intent (specified model, agent does not observe U)
            #I = intent(U_samples[0, i], U_samples[1, i])
            I = np.random.randint(config.K)
            # U_index = I  # Choose from columns 0-3 in theta ()

            if data_type == 'observational':
                action = I
            if data_type == 'experimental':
                action = np.random.choice(config.K)

            win_prob = theta[action, I]  # Probability of success
            reward = np.random.choice(
                2, p=[1 - win_prob, win_prob])  # Pull arm
            samples[action, reward] += 1

    else:

        for i in range(config.K):
            if data_type == 'observational':
                samples[i, :] = [(1 - theta[i, i]) * config.SAMPLES_PER_ARM, theta[i, i] * config.SAMPLES_PER_ARM]
            if data_type == 'experimental':
                samples[i, :] = [(1 - np.mean(theta[i, :])) * config.SAMPLES_PER_ARM, np.mean(theta[i, :]) * config.SAMPLES_PER_ARM]

    return samples

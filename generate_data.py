# from datasim import create_data, create_random_data
# from config import Config
# import numpy as np
# import sys
#
# config = Config()
# # ---------------------------------------------------------------------- #
# #  Simulate experimental and observational data (if applied in settings) #
# # ---------------------------------------------------------------------- #
#
# 
# def generate_and_save_data():
#     if config.SIMULATE_OBS_EXP_DATA:
#         data_exp, data_obs = create_data()
#         np.save('data_exp', data_exp)
#         np.save('data_obs', data_obs)
#         # if config.CREATE_OBS_EXP_DATA_ONLY:
#         #     sys.exit()
#
#
# def generate_and_save_n_data(N):
#     if config.USE_RANDOM_DATA:
#         data_exp_list, data_obs_list, theta_list = create_random_data(N)
#         np.save('data_exp_list', data_exp_list)
#         np.save('data_obs_list', data_obs_list)
#         np.save('data_exp_list', theta_list)
#
# # data_exp_list, data_obs_list, theta_list = create_random_data()
# # np.save('data_exp_list', data_exp_list)
# # np.save('data_obs_list', data_obs_list)
# # np.save('data_exp_list', theta_list)

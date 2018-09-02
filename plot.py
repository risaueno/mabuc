#%%
import numpy as np
import pymc3 as pm
import theano.tensor as tt
import matplotlib.pyplot as plt
from helper import tic, toc, smooth
from scipy.stats import beta
from theano.printing import pydotprint
import os

from config import Config
config = Config()

#plot_traceplot = False
#run_gradient_descent = False

#import datetime
#print(str(datetime.datetime.now()).split('.')[0])

#%%

# TO FILL IN STD, DO THIS!
# plt.fill_between(mstd.index, ma-2*mstd, ma+2*mstd, color='b', alpha=0.2)

datetime = '2018-09-02-19-54-57'
N = 99

save_dir = './data_folder/vanilla/'
save_dir = './data_folder/ihdp/'

algs = ['IVWA - TS++', 'IVWA - TS+', 'IVWA - TS', 'IVWA - paper++', 'IVWA - paper+', 'IVWA - paper']
algs = ['IVWA - paper++', 'IVWA - paper+', 'IVWA - paper']

#%%

plt.figure()

for i in range(len(algs)):

    pprob_string = save_dir + datetime + '_N=' + str(N) + '_AVE_PProb_' + algs[i] + '.npy'
    pprob = np.load(pprob_string)

    regret_string = save_dir + datetime + '_N=' + str(N) + '_AVE_Regret_' + algs[i] + '.npy'
    regret = np.load(regret_string)

    cumregret_string = save_dir + datetime + '_N=' + str(N) + '_AVE_CumRegret_' + algs[i] + '.npy'
    cumregret = np.load(cumregret_string)

    accuracy_string = save_dir + datetime + '_N=' + str(N) + '_AVE_Accuracy_' + algs[i] + '.npy'
    accuracy = np.load(accuracy_string)

    ax = plt.subplot(221)
    ax.plot(smooth(pprob), label=algs[i])

    ax = plt.subplot(222)
    ax.plot(smooth(regret), label=algs[i])

    ax = plt.subplot(223)
    ax.plot(smooth(cumregret), label=algs[i])

    ax = plt.subplot(224)
    ax.plot(smooth(accuracy), label=algs[i])


plt.subplot(221)
plt.xlabel('t')
plt.ylabel('Probability of optimal action')

plt.subplot(222)
plt.xlabel('t')
plt.ylabel('Current Regret')

plt.subplot(223)
plt.xlabel('t')
plt.ylabel('Total Regret')

plt.subplot(224)
plt.xlabel('t')
plt.ylabel('Accuracy')

plt.legend()
plt.show()

#%%

PProb_file1 = 'IHDP_PProb_RDC*_edit__' + date + ' ' + time + '.npy'
Regret_file1 = 'IHDP_Regret_RDC*_edit__' + date + ' ' + time + '.npy'
PProb_file2 = 'IHDP_PProb_RDC_original++_' + date + ' ' + time + '.npy'
Regret_file2 = 'IHDP_Regret_RDC_original++_' + date + ' ' + time + '.npy'
PProb_file3 = 'IHDP_PProb_RDC_original++_' + date + ' ' + time + '.npy'
Regret_file3 = 'IHDP_Regret_RDC_original++_' + date + ' ' + time + '.npy'

PProb1 = np.load(os.path.join(save_dir, PProb_file1))
Regret1 = np.load(os.path.join(save_dir, Regret_file1))
PProb2 = np.load(os.path.join(save_dir, PProb_file2))
Regret2 = np.load(os.path.join(save_dir, Regret_file2))
PProb3 = np.load(os.path.join(save_dir, PProb_file3))
Regret3 = np.load(os.path.join(save_dir, Regret_file3))

plt.figure()

max_x_range = 200

ax = plt.subplot(211)
ax.plot(smooth(PProb), label='1')
ax.plot(smooth(PProb2), label='2')
ax.plot(smooth(PProb3), label='3')
# ax.set_yticks(np.arange(0, 1., 0.1))
# ax.set_xticks(np.arange(0, max_x_range, 10))
ax.grid()

ax2 = plt.subplot(212)
ax2.plot(smooth(Regret), label='1')
ax2.plot(smooth(Regret2), label='2')
ax2.plot(smooth(Regret3), label='3')
# ax2.set_yticks(np.arange(0, 100, 10))
# ax2.set_xticks(np.arange(0, max_x_range, 10))
ax2.grid()

plt.subplot(211)
plt.xlabel('t')
plt.ylabel('Probability of optimal action')
plt.xlim(0, len(PProb))
# plt.ylim(0.5, 1)
plt.legend()

plt.subplot(212)
plt.xlabel('t')
plt.ylabel('Regret')
plt.legend()
plt.fill_between(np.arange(len(Regret2)), smooth(Regret2) - smooth(Regret), smooth(Regret2) + smooth(Regret3), color='b', alpha=0.1)
plt.xlim(0, len(Regret))
# plt.ylim(0, 1)

plt.show()

'''
MCMC
'''
import numpy as np
from scipy.misc import comb
from itertools import product
import pandas as pd
import copy
import json
import timeit
import scipy.stats as stats
import os
import netSTI.net as net
import netSTI.adaptMCMC as mcmc
os.chdir("/Users/szu-yukao/Documents/Network_structure_and_STI/networkSTI")
cwd = os.getcwd()
print(cwd)

import warnings
warnings.filterwarnings("ignore")

nchain = 1

pars = [14]
pars_name = ["meanActs"]
pars_lb = [10]
pars_ub = [100]

scales = {"meanActs": 0.005}

target_I = np.repeat(0.05, 260)
sd_I = target_I * 0.1

outcomes = mcmc.AdpativeMCMC(iters = 100, burn_in = 10, adapt_par = [10, 10], \
	pars = pars, pars_name = pars_name, pars_lb = pars_lb, pars_ub = pars_ub, scales = scales, \
	target = target_I, target_sd = sd_I, verbose = 1, function = net.SIR_net_generator, \
	run = 1, Npop = 5000, years = 50, days = 14, \
	strategy = "PN", graph = "random", independent = True, calibration = True, analysis_window = 20)


with open('cal_data/mcmc_results_' + str(nchain) + '.txt', 'w') as fout:
    json.dump(outcomes, fout)


import matplotlib as mpl
print(mpl.rcParams['backend'])
mpl.use('TkAgg')
import matplotlib.pyplot as plt
plt.get_backend()
import corner

with open('data/mcmc_results_'+str(nchain)+'.txt') as json_file:  
    outcomes = json.load(json_file)

samples = np.array(outcomes['meanActs'])

plt.plot(samples)
plt.show()


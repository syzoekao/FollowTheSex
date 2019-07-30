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

nchain = 1

pars = [29]
pars_name = ["meanActs"]
pars_lb = [10]
pars_ub = [100]

scales = {"meanActs": 0.1}

target_cum_dx = np.repeat(0.14, 9)
sd_cum_dx = target_cum_dx * 0.1

outcomes = mcmc.AdpativeMCMC(iters = 1000, burn_in = 100, adapt_par = [100, 100], \
	pars = pars, pars_name = pars_name, pars_lb = pars_lb, pars_ub = pars_ub, scales = scales, \
	target = target_cum_dx, target_sd = sd_cum_dx, verbose = 1, function = net.SIR_net_generator, \
	run = 1, Npop = 5000, years = 20, days = 14, \
	strategy = "null", graph = "random", adjust_sex_acts = False, calibration = True)

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


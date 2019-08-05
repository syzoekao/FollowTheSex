import time
import numpy as np
import os
import json
import sys
import netSTI.net as net
import scipy.stats as stats

import os
os.chdir("/panfs/roc/groups/0/ennse/kaoxx085/networkSTI")
cwd = os.getcwd()
print(cwd)

import warnings
warnings.filterwarnings("ignore")

def calibration_task(g, n_yr, a_wind, indep, tar_prev, par_mean, par_sd): 
	# g = "random"
	# n_yr = 50
	# a_wind = 20
	# indep = True
	target = np.repeat(tar_prev, 260)
	target_sd = np.repeat(tar_prev * 0.1, 260)

	par = np.random.normal(par_mean, par_sd)
	prior = stats.norm.pdf(par, loc = par_mean, scale = par_sd)

	sim_out = net.SIR_net_generator(meanActs = par, run = 1, Npop = 5000, years = n_yr, days = 14, 
	strategy = "PN", graph = g, independent = indep, calibration = True, 
	analysis_window = a_wind)

	lk = np.sum(stats.norm.pdf(sim_out, loc = target, scale = target_sd))

	weight = lk / prior 

	return [par, prior, lk, weight]


n_sim = 1000

g = "random"
n_yr = 50
a_wind = 20
indep = True
tar_prev = 0.05
par_mean = 14
par_sd = 2


result = [None] * n_sim

for i in range(n_sim): 
	tmp = calibration_task(g, n_yr, a_wind, indep, tar_prev, par_mean, par_sd)
	result[i] = tmp

with open('results/SIRresults/randomLowUncorr1.txt', 'w') as fout:
	json.dump(result, fout)




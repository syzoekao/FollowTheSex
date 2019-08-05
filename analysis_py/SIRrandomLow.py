import time
import numpy as np
import os
import json
import sys
from celery import Celery
from celery.result import ResultSet
import celery.signals
import netSTI.net as net
import scipy.stats as stats

import warnings
warnings.filterwarnings("ignore")

app = Celery('SIRrandomLow',
	broker='redis://localhost:6379/0',
	backend='redis://')
 
@celery.signals.worker_process_init.connect()
def seed_rng(**_):
	"""
	Seeds the numpy random number generator.
	"""
	np.random.seed()

@app.task
def sim_task(run): 
	g = "random"
	n_yr = 50
	a_wind = 20
	indep = True
	target = np.repeat(0.05, 260)
	target_sd = np.repeat(0.05 * 0.1, 260)

	mean = 14
	sd = 1
	par = np.random.normal(mean, sd)

	prior = stats.norm.pdf(par, loc = mean, scale = sd)

	sim_out = net.SIR_net_generator(meanActs = par, run = run, Npop = 5000, years = n_yr, days = 14, 
	strategy = "PN", graph = g, independent = indep, calibration = True, 
	analysis_window = a_wind)

	lk = np.sum(stats.norm.pdf(sim_out, loc = target, scale = target_sd))

	weight = lk / prior 

	return [par, prior, lk, weight]


if __name__ == "__main__":
	os.chdir("/Users/szu-yukao/Documents/Network_structure_and_STI/networkSTI")
	cwd = os.getcwd()
	print(cwd)

	start_time = time.time()
	ret = ResultSet([sim_task.delay(i) for i in range(500)])
	print(len(ret.get()))
	end_time = time.time()
	print("CeleryTime:", end_time - start_time)

	with open('results/SIRresults/random_unadjusted_low.txt', 'w') as fout:
		json.dump(ret.get(), fout)

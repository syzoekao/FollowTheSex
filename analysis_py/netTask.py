import time
import numpy as np
import os
import json
import sys
from celery import Celery
from celery.result import ResultSet
import celery.signals
import netSTI.net as net
import netSTI.posterior as post

import os
os.chdir("/Users/szu-yukao/Documents/Network_structure_and_STI/networkSTI")
cwd = os.getcwd()
print(cwd)


app = Celery('netTask',
             broker='redis://localhost:6379/0',
             backend='redis://')
 
@celery.signals.worker_process_init.connect()
def seed_rng(**_):
    """
    Seeds the numpy random number generator.
    """
    np.random.seed()

@app.task
def sim_task(run, indep, graph): 
    return net.SIR_net_generator(run, Npop = 5000, years = 5, days = 14, 
    graph = graph, pEFoI = (1 / 5000) / 2 , 
    pContact_PN = 0.49, pContact_ept = 0.7, pContact_tr = 0.7, 
    p_treat_PN = 0.71, p_treat_ept = 0.79, p_treat_tr = 0.79, 
    independent = indep, calibration = False, 
    analysis_window = 2, output_net = True)


if __name__ == "__main__":
    os.chdir("/Users/szu-yukao/Documents/Network_structure_and_STI/networkSTI")
    cwd = os.getcwd()
    print(cwd)

    graph = "power_law"
    indep = True
    n_samp = 500

    if indep == True: 
        cor_or_not = "Uncorr"
    else: 
        cor_or_not = "Corr"

    # n_samp = 100
    # par_vec = post.SamplePosterior(g = graph, target_prev = 0.20, indep = indep).sample_from_posterior(n_samp)
    # par_vec = par_vec.tolist()

    start_time = time.time()
    ret = ResultSet([sim_task.delay(i, indep, graph) for i in range(n_samp)])
    print(len(ret.get()))
    end_time = time.time()
    print("CeleryTime:", end_time - start_time)

    with open('results/netout/netout_' + graph + cor_or_not + '.txt', 'w') as fout:
        json.dump(ret.get(), fout)

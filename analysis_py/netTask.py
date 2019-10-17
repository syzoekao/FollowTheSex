import timeit
import time
import numpy as np
import os
import json
import sys
from celery import Celery
from celery.result import ResultSet
import celery.signals
import netSTI.netSTI as netSTI

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
def sim_task(run, indep, graph, pEFoI): 
    return netSTI.SIR_net_generator(run, 5000, years = 5, days = 14, 
    graph = graph, pEFoI = pEFoI, independent = indep, PrEPuser = True, base_case = False, 
    trend = False, analysis_window = 2, output_netsum = False, output_net = False)


if __name__ == "__main__":
    os.chdir("/Users/szu-yukao/Documents/Network_structure_and_STI/networkSTI")
    cwd = os.getcwd()
    print(cwd)

    graph = "power_law"
    indep = False
    pEFoI = (1 / 5000) / 2
    n_samp = 500

    if indep == True: 
        cor_or_not = "Uncorr"
    else: 
        cor_or_not = "Corr"

    if pEFoI == (1 / 5000) / 2: 
        lvl = "Low"
    else: 
        lvl = "High"


    start_time = time.time()
    ret = ResultSet([sim_task.delay(i, indep, graph, pEFoI) for i in range(n_samp)])
    print(len(ret.get()))
    end_time = time.time()
    print("CeleryTime:", end_time - start_time)

    with open('results/RRresults/trend/' + graph + cor_or_not + lvl + '.txt', 'w') as fout:
        json.dump(ret.get(), fout)

    # with open('results/netout/netout_' + graph + cor_or_not + '.txt', 'w') as fout:
    #     json.dump(ret.get(), fout)

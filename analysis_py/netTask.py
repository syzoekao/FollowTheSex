import time
import netSTI.net as net
import numpy as np
import os
import json
import sys
from celery import Celery
from celery.result import ResultSet
import celery.signals

 
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
def sim_task(run): 
    '''
    graph_function = random_graph_generator
    graph_function = community_graph_generator
    graph_function = powerLaw_graph_generator
    '''
    graph_function = net.random_graph_generator
    return net.SIR_net_generator(run, graph_function, 
        Npop = 5000, Ndegree = 4*10, time_horizon = 12*10, 
        pInf = 0.135, pCondom = 0.44, redCondom = 0.6, 
        durI = 6, rScr = 0.3/12, pContact_tr = 0.7, pContact_ept = 0.7, 
        pContact_PN = 0.49, p_treat_tr = 0.79, p_treat_ept = 0.79, p_treat_PN = 0.71, 
        init_prev = 0.1, n_cluster = 5, strategy = "null", max_contact = 30, 
        max_ept = 500, alpha = 1, check_steady_state = True)


if __name__ == "__main__":
    os.chdir("/Users/szu-yukao/Documents/Network_structure_and_STI/networkSTI")
    cwd = os.getcwd()
    print(cwd)

    start_time = time.time()
    ret = ResultSet([sim_task.delay(i) for i in range(10)])
    print(len(ret.get()))
    end_time = time.time()
    print("CeleryTime:", end_time - start_time)

    with open('results/trend_random.txt', 'w') as fout:
        json.dump(ret.get(), fout)



'''
if __name__ == "__main__":
    start_time = time.time()
    
    n_runs = [x for x in range(1, 50, 1)]
    # Using `delay` runs the task async
    rs = ResultSet([simulation_task.delay(run) for run in n_runs])
     
    # Wait for the tasks to finish
    rs.get()
 
    end_time = time.time()
 
    print("CelerySquirrel:", end_time - start_time)
    # CelerySquirrel: 2.4979639053344727

    rs.get()
    print(rs.get()[0])

    results = rs.get()


if __name__ == "__main__":
    os.chdir("/Users/szu-yukao/Documents/Network_structure_and_STI/networkSTI")
    # os.chdir("/panfs/roc/groups/0/ennse/kaoxx085/diss")
    cwd = os.getcwd()
    print(cwd)

    sys.stdout = open('Console_out', 'w')

    n_runs = [x for x in range(1, 11, 1)]

    start_time = time.time()
    result = ResultSet([simulation_task.delay(run) for run in n_runs])
    end_time = time.time()
     
    print("CelerySquirrel:", end_time - start_time)

    result.get()
    result_l = result.get()

    import json
    with open('results/outputfile', 'w') as fout:
        json.dump(results_l, fout)

    sys.stdout.close()
'''




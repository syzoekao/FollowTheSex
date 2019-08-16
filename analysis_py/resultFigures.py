'''
Prevalence trends
'''

import seaborn as sbn 
import numpy as np
import pandas as pd
import json
import matplotlib as mpl
from matplotlib import collections  as mc
from matplotlib.patches import Rectangle
import copy
print(mpl.rcParams['backend'])
mpl.use('TkAgg')
import matplotlib.pyplot as plt
plt.get_backend()
import os
os.chdir("/Users/szu-yukao/Documents/Network_structure_and_STI/networkSTI")
cwd = os.getcwd()
print(cwd)

graph = ['random', 'community', 'power_law', 'empirical']
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

for g in graph: 
    with open('results/RRresults/trend/' + g + cor_or_not + lvl + '.txt') as json_file:  
        temp = json.load(json_file)

    policy_ls = [k for k, v in temp[0].items()][1:]

    null = np.zeros((len(temp), len(temp[0]['null'])))
    pn = np.zeros((len(temp), len(temp[0]['null'])))
    ept = np.zeros((len(temp), len(temp[0]['null'])))
    trace = np.zeros((len(temp), len(temp[0]['null'])))

    for i in range(len(temp)): 
        null[i] = temp[i]['null']
        pn[i] = temp[i]['pn']
        ept[i] = temp[i]['ept']
        trace[i] = temp[i]['tracing']

    fig = plt.figure(figsize=(7, 5))
    m = np.mean(null, axis = 0)
    ub = np.percentile(null, 95, axis = 0)
    lb = np.percentile(null, 5, axis = 0)
    x_data = np.arange(null.shape[1])
    plt.plot(x_data, ub, color = 'gray', linewidth = 0, alpha = 0.7)
    plt.plot(x_data, lb, color = 'gray', linewidth = 0, alpha = 0.7)
    plt.plot(x_data, m, color = 'gray', linewidth = 2)
    plt.fill_between(x_data, lb, ub, color = 'gray', alpha = 0.7)


    m = np.mean(pn, axis = 0)
    ub = np.percentile(pn, 95, axis = 0)
    lb = np.percentile(pn, 5, axis = 0)
    x_data = np.arange(pn.shape[1])
    plt.plot(x_data, ub, color = 'royalblue', linewidth = 0, alpha = 0.7)
    plt.plot(x_data, lb, color = 'royalblue', linewidth = 0, alpha = 0.7)
    plt.plot(x_data, m, color = 'royalblue', linewidth = 2)
    plt.fill_between(x_data, lb, ub, color = 'royalblue', alpha = 0.7)


    m = np.mean(ept, axis = 0)
    ub = np.percentile(ept, 95, axis = 0)
    lb = np.percentile(ept, 5, axis = 0)
    x_data = np.arange(ept.shape[1])
    plt.plot(x_data, ub, color = 'limegreen', linewidth = 0, alpha = 0.7)
    plt.plot(x_data, lb, color = 'limegreen', linewidth = 0, alpha = 0.7)
    plt.plot(x_data, m, color = 'limegreen', linewidth = 2)
    plt.fill_between(x_data, lb, ub, color = 'limegreen', alpha = 0.7)


    m = np.mean(trace, axis = 0)
    ub = np.percentile(trace, 95, axis = 0)
    lb = np.percentile(trace, 5, axis = 0)
    x_data = np.arange(trace.shape[1])
    plt.plot(x_data, ub, color = 'salmon', linewidth = 0, alpha = 0.7)
    plt.plot(x_data, lb, color = 'salmon', linewidth = 0, alpha = 0.7)
    plt.plot(x_data, m, color = 'salmon', linewidth = 2)
    plt.fill_between(x_data, lb, ub, color = 'salmon', alpha = 0.7)
    plt.ylim([0, 0.25])

    plt.title(g)
    color_ls = ["gray", "royalblue", "limegreen", "salmon"]
    handles = [Rectangle((0, 0), 1, 1, color = c, ec = c) for c in color_ls] 
    labels = ["Null", "PN", "EPT", "Tracing"] 
    plt.legend(handles, labels, loc = 2, frameon = False, fontsize = 'small')

    plt.xlabel("time step")
    plt.ylabel("prevalence")
    plt.text(0, 0.01, 'Mean prevalence at last time step\n' + \
        'null: ' + str(np.round(np.mean(np.mean(null, axis = 0)[-1]), 2)) + 
        '\nPN: ' + str(np.round(np.mean(np.mean(pn, axis = 0)[-1]), 2)) + 
        '\nEPT: ' + str(np.round(np.mean(np.mean(ept, axis = 0)[-1]), 2)) + 
        '\ntracing: ' + str(np.round(np.mean(np.mean(trace, axis = 0)[-1]), 2)), fontsize = 'x-small')
    plt.tight_layout()
    plt.savefig('results/RRresults/trend/' + g + '_' + cor_or_not + lvl + '.png', dpi = 400)
    plt.clf()



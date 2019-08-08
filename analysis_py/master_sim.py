'''
Check degree distribution of power law
'''
import numpy as np
import netSTI.net as net
import netSTI.mydata as data
import pandas as pd
import json
import matplotlib as mpl
from sklearn.linear_model import LinearRegression
import copy
from pylab import *
print(mpl.rcParams['backend'])
mpl.use('TkAgg')
import matplotlib.pyplot as plt
plt.get_backend()
import os
os.chdir("/Users/szu-yukao/Documents/Network_structure_and_STI/networkSTI")
cwd = os.getcwd()
print(cwd)

Npop = 5000
ID = np.arange(Npop)
dur = data.SexBehavior().dur
dur_dist = data.SexBehavior().dur_dist
years = 20
Ndegree = 4 * years
days = 14
unit_per_year = int(365 / days)
time_horizon = unit_per_year * years

nsamp = 100
fit_vec = np.zeros(nsamp)
avg_deg = np.zeros(nsamp)

fig = plt.figure(figsize=(6,4))
ax = plt.gca()
for run in range(nsamp): 
    print(run)
    rel_hist = net.power_law_graph_generator(ID, Npop, dur, dur_dist, time_horizon, n_yrs = years)

    tmp_rel = rel_hist[(rel_hist[:, 3] >= (10 * unit_per_year))]
    tmp_names, tmp_degree = np.unique(tmp_rel[:, :2], return_counts = True)
    degree, counts = np.unique(tmp_degree, return_counts = True)
    degree_dist = counts / np.sum(counts)
    fit = np.polyfit(np.log(degree), np.log(degree_dist), 1)
    fit_vec[run] = fit[1]
    fit_fn = np.poly1d(fit)
    avg_deg[run] = np.mean(tmp_degree)
    ax.scatter(np.log(degree), np.log(degree_dist), edgecolor = 'limegreen', facecolors='none')
    ax.plot(np.log(degree), fit_fn(np.log(degree)), 'royalblue')

x_ticks = np.array([0.1, 1, 2, 3, 4, 5])
y_ticks = np.array([-8, -6, -4, -2, -0.1])
ax.set_xticks(x_ticks)
ax.set_xticklabels(np.round(np.exp(x_ticks)))
ax.set_yticks(y_ticks)
ax.set_yticklabels(np.round(np.exp(y_ticks), 3))
ax.text(0.1, -7, 'mean of the slope is\n' + str(np.round(np.mean(fit_vec), 2)) + \
    '\naverage mean degree\n(last 10 years):\n' + str(np.round(np.mean(avg_deg), 2)), fontsize=12)
plt.xlabel('degree')
plt.ylabel('proportion')
plt.title('100 network samples')
plt.savefig('results/power law degree distribution.eps', format='eps', dpi=1000)


from scipy.stats import poisson

Npop = 5000
ID = np.arange(Npop)
dur = data.SexBehavior().dur
dur_dist = data.SexBehavior().dur_dist
years = 20
Ndegree = 4 * years
days = 14
unit_per_year = int(365 / days)
time_horizon = unit_per_year * years
n_cluster = 5
ID_cluster = (np.floor(ID / (Npop/n_cluster)) + 1).astype(int)

nsamp = 100
avg_deg = np.zeros(nsamp)

fig = plt.figure(figsize=(6,4))
ax = plt.gca()
for run in range(nsamp):
    print(run)
    rel_hist = net.random_graph_generator(ID, Npop, Ndegree, dur, dur_dist, time_horizon)

    tmp_rel = rel_hist[(rel_hist[:, 3] >= (10 * unit_per_year))]
    tmp_names, tmp_degree = np.unique(tmp_rel[:, :2], return_counts = True)
    ax.hist(tmp_degree, density=True, color = 'limegreen')
    x_pois = np.arange(np.min(tmp_degree), np.max(tmp_degree))
    pois = poisson.pmf(x_pois, np.mean(tmp_degree))
    avg_deg[run] = np.mean(tmp_degree)
    ax.plot(x_pois, pois, 'royalblue')

ax.text(5, 0.05, 'average mean degree\n(last 10 years):\n' + \
    str(np.round(np.mean(avg_deg), 2)), fontsize=12)
plt.xlabel('degree')
plt.ylabel('proportion')
plt.title('100 network samples')
plt.savefig('results/random degree distribution.eps', format='eps', dpi=1000)


'''
Average degree per time step
'''

all_deg = [None] * 3
net_name = ["random", "community", "power_law"]
for x in range(len(net_name)): 
    with open('results/trend/trend_' + net_name[x] + '.txt') as json_file:  
        temp = json.load(json_file)

    temp_ls = [None] * len(temp)
    for i in range(len(temp)): 
        temp_ls[i] = temp[i]["avg_degree"]

    all_deg[x] = np.array(temp_ls)

cols = ['royalblue', 'limegreen', 'tomato']
fig = plt.figure(figsize=(6,4))
for run in range(len(all_deg)): 
    plt.plot(np.mean(all_deg[run], axis = 0), cols[run], label = net_name[run])
plt.xlabel('time step')
plt.ylabel('degree')
plt.legend(loc= 'lower right')
plt.title('Average degree at each time step\n across 50 simulations')
plt.savefig('results/average degree comparison.eps', format='eps', dpi=1000)



'''
Trends
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
pEFoI = (1 / 5000) / 2 * 10
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
    with open('results/RRresults/trend/' + g + '_' + cor_or_not + lvl + '.txt') as json_file:  
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

    plt.title(g + ' ' + cor_or_not + '_' + lvl)
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




'''
Combine every simulation (5000)
'''

import seaborn as sbn 
import numpy as np
import pandas as pd
import json
import matplotlib as mpl
import copy
print(mpl.rcParams['backend'])
mpl.use('TkAgg')
import matplotlib.pyplot as plt
plt.get_backend()
import os
os.chdir("/Users/szu-yukao/Documents/Network_structure_and_STI/networkSTI")
cwd = os.getcwd()
print(cwd)

results_key = ['powerLaw_graph', 'random_graph', 'community_graph']
strategy_key = ['null', 'PN', 'EPT', 'contact tracing_degree']


def read_json_file_null(x): 
    col_order = ['run', 'strategy', 'p_treat_ept', 'p_treat_PN', 'p_treat_tr', 
    'avg_durI', 'prevalence', 'newI', 'newT', 'nInfTreatIntervention', 
    'nInvestigate', 'nNotified', 'nDeliver', 'nOvertreat', 'ever_infected_node', 'treat_per_intervention', 'avg_num_inf10', 
    'CostMedicine', 'CostTracing', 'CostTest', 'TotalCost', 'Util']

    temp_ls = []
    
    for file_n in range(1, 51): 
        with open('results/high notification/'+ x + '_null_' + str(file_n)+'.txt') as json_file:  
            temp_ls += json.load(json_file)
    
    out_dict = dict()

    res_key = ['avg_durI', 'prevalence', 'newI', 'newT', 'nInfTreatIntervention', 
        'nInvestigate', 'nNotified', 'nDeliver', 'nOvertreat', 'ever_infected_node', 'treat_per_intervention', 'avg_num_inf10', 
        'CostMedicine', 'CostTracing', 'CostTest', 'TotalCost', 'Util']

    out_dict['run'] = [temp_ls[x]['run'] for x in range(0, len(temp_ls))]
    out_dict['strategy'] = ['null' for x in range(0, len(temp_ls))]

    for k in res_key: 
        out_dict[k] = [temp_ls[x]['summary'][k] for x in range(0, len(temp_ls))]

    out_dict['p_treat_ept'] = [temp_ls[x]['p_treat_ept'] for x in range(0, len(temp_ls))]
    out_dict['p_treat_PN'] = [temp_ls[x]['p_treat_PN'] for x in range(0, len(temp_ls))]
    out_dict['p_treat_tr'] = [temp_ls[x]['p_treat_tr'] for x in range(0, len(temp_ls))]

    outDF = pd.DataFrame(out_dict)
    outDF = outDF[col_order]

    return outDF


def read_json_file_strategy(x, strategy): 
    out_ls = dict()
    col_order = ['run', 'strategy', 'p_treat_ept', 'p_treat_PN', 'p_treat_tr', 
    'avg_durI', 'prevalence', 'newI', 'newT', 'nInfTreatIntervention', 
    'nInvestigate', 'nNotified', 'nDeliver', 'nOvertreat', 'ever_infected_node', 'treat_per_intervention', 'avg_num_inf10', 
    'CostMedicine', 'CostTracing', 'CostTest', 'TotalCost', 'Util']

    for pr in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.71, 0.79, 0.8, 0.9, 1]:
        temp_ls = []
        for file_n in range(1, 51): 
            try: 
                with open('results/high notification/'+ x + '_' + strategy + '_' + str(pr) + '_' + str(file_n)+'.txt') as json_file:  
                    temp_ls += json.load(json_file)
            except FileNotFoundError:
                print("graph = " + x + "; strategy = " + strategy + "; pr = " + str(pr) + "; file_n = " + str(file_n))
        
        out_dict = dict()

        res_key = ['avg_durI', 'prevalence', 'newI', 'newT', 'nInfTreatIntervention', 
            'nInvestigate', 'nNotified', 'nDeliver', 'nOvertreat', 'ever_infected_node', 'treat_per_intervention', 'avg_num_inf10', 
            'CostMedicine', 'CostTracing', 'CostTest', 'TotalCost', 'Util']

        out_dict['run'] = [temp_ls[x]['run'] for x in range(0, len(temp_ls))]
        out_dict['strategy'] = [strategy for x in range(0, len(temp_ls))]

        for k in res_key: 
            out_dict[k] = [temp_ls[x]['summary'][k] for x in range(0, len(temp_ls))]

        out_dict['p_treat_ept'] = [temp_ls[x]['p_treat_ept'] for x in range(0, len(temp_ls))]
        out_dict['p_treat_PN'] = [temp_ls[x]['p_treat_PN'] for x in range(0, len(temp_ls))]
        out_dict['p_treat_tr'] = [temp_ls[x]['p_treat_tr'] for x in range(0, len(temp_ls))]

        outDF = pd.DataFrame(out_dict)
        outDF = outDF[col_order]
        out_ls[str(pr)] = outDF

    return out_ls


powerLawDF = read_json_file_null('powerLaw_graph')
for strategy in ['PN', 'EPT', 'contact tracing_degree']: 
    tmpDF = read_json_file_strategy('powerLaw_graph', strategy)
    keys = [k for k, v in tmpDF.items()]
    for k in keys: 
        powerLawDF = powerLawDF.append(tmpDF[k])


randomDF = read_json_file_null('random_graph')
for strategy in ['PN', 'EPT', 'contact tracing_degree']: 
    tmpDF = read_json_file_strategy('random_graph', strategy)
    keys = [k for k, v in tmpDF.items()]
    for k in keys: 
        randomDF = randomDF.append(tmpDF[k])


communityDF = read_json_file_null('community_graph')
for strategy in ['PN', 'EPT', 'contact tracing_degree']: 
    tmpDF = read_json_file_strategy('community_graph', strategy)
    keys = [k for k, v in tmpDF.items()]
    for k in keys: 
        communityDF = communityDF.append(tmpDF[k])


powerLawDF.to_csv('results/high notification/powerLawDF.csv')
randomDF.to_csv('results/high notification/randomDF.csv')
communityDF.to_csv('results/high notification/communityDF.csv')




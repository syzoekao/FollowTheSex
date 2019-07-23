'''
Trends
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

x = "community"
with open('results/trend/trend_' + x + '.txt') as json_file:  
    temp_ls = json.load(json_file)

ret = np.array(temp_ls)
print(np.mean(ret[:, 200:], axis = 0))
fig = plt.figure(figsize=(6,4))
plt.plot(ret[0], color = 'limegreen', linewidth=2)
plt.title("0.033")
for i in range(1, ret.shape[0]): 
    plt.plot(ret[i], color = 'limegreen', linewidth=2)
plt.tight_layout()
plt.savefig('results/trend/trend' + x + '.eps', format='eps', dpi=1000)




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




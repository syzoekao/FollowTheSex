import numpy as np
import netSTI.netSTI as netSTI
import pandas as pd
import json
import matplotlib as mpl
from sklearn.linear_model import LinearRegression
import copy

import os
os.chdir("/Users/szu-yukao/Documents/Network_structure_and_STI/networkSTI")
cwd = os.getcwd()
print(cwd)


def get_sim_result(graph, lvl, cor_or_not, texts = ''): 
	'''
	graph = 'random'
	cor_or_not = 'Uncorr'
	lvl = 'Low'
	'''
	temp_ls = []

	for file_n in range(1, 51): 
		try: 
			with open('results/RR2results/'+ graph + cor_or_not + lvl + '_' + str(file_n)+'.txt') as json_file:  
				temp_ls += json.load(json_file)
		except: 
			print(graph + cor_or_not + lvl + '_' + str(file_n))

	key_lv1 = [k for k, v in temp_ls[0].items()]
	key_lv2 = [k for k, v in temp_ls[0]['null'].items()]
	pn_key_lv = [k for k, v in temp_ls[0]['pn'].items()]
	ept_key_lv = [k for k, v in temp_ls[0]['ept'].items()]
	tr_key_lv = [k for k, v in temp_ls[0]['tracing'].items()]

	# null
	tmp_null = [None] * len(temp_ls)
	for i in range(len(temp_ls)): 
		tmp_null[i] = [v for k, v in temp_ls[i]['null'].items()]
	tmp_null = np.array(tmp_null)
	tmp_null = pd.DataFrame(tmp_null)
	tmp_null.columns = key_lv2
	tmp_null['p_treat_pn'] = 0
	tmp_null['p_treat_ept'] = 0
	tmp_null['p_treat_tr'] = 0
	tmp_null['strategy'] = 'screen'

	# PN
	tmp_pn = [None] * len(pn_key_lv)
	for v in range(len(pn_key_lv)): 
		tmp_ret = [None] * len(temp_ls)
		for i in range(len(temp_ls)): 
			tmp_ret[i] = [v for k, v in temp_ls[i]['pn'][pn_key_lv[v]].items()]
		tmp_ret = pd.DataFrame(np.array(tmp_ret))
		tmp_ret.columns = key_lv2
		tmp_ret['p_treat_pn'] = pn_key_lv[v]
		tmp_ret['p_treat_ept'] = 0
		tmp_ret['p_treat_tr'] = 0	
		tmp_pn[v] = tmp_ret
	tmp_pn = pd.concat(tmp_pn)
	tmp_pn['strategy'] = 'pn'

	# EPT
	tmp_ept = [None] * len(ept_key_lv)
	for v in range(len(ept_key_lv)): 
		tmp_ret = [None] * len(temp_ls)
		for i in range(len(temp_ls)): 
			tmp_ret[i] = [v for k, v in temp_ls[i]['ept'][ept_key_lv[v]].items()]
		tmp_ret = pd.DataFrame(np.array(tmp_ret))
		tmp_ret.columns = key_lv2
		tmp_ret['p_treat_pn'] = 0
		tmp_ret['p_treat_ept'] = ept_key_lv[v]
		tmp_ret['p_treat_tr'] = 0	
		tmp_ept[v] = tmp_ret
	tmp_ept = pd.concat(tmp_ept)
	tmp_ept['strategy'] = 'ept'

	# tracing
	tmp_tr = [None] * len(tr_key_lv)
	for v in range(len(tr_key_lv)): 
		tmp_ret = [None] * len(temp_ls)
		for i in range(len(temp_ls)): 
			tmp_ret[i] = [v for k, v in temp_ls[i]['tracing'][tr_key_lv[v]].items()]
		tmp_ret = pd.DataFrame(np.array(tmp_ret))
		tmp_ret.columns = key_lv2
		tmp_ret['p_treat_pn'] = 0
		tmp_ret['p_treat_ept'] = 0
		tmp_ret['p_treat_tr'] = tr_key_lv[v]
		tmp_tr[v] = tmp_ret
	tmp_tr = pd.concat(tmp_tr)
	tmp_tr['strategy'] = 'trace'

	out_file = pd.concat((tmp_null, tmp_pn, tmp_ept, tmp_tr))
	file_name = graph + cor_or_not + lvl + texts

	return out_file, file_name


txt = '(corr_scr 2yrs)'

graph_key = ['random', 'community', 'power_law', 'empirical']


for cor_or_not in ['Corr']: # "Uncorr"
	for lvl in ['Low', 'High']: 
		for graph in graph_key: 
			tmp_df, file_name = get_sim_result(graph, lvl, cor_or_not, texts = txt)
			# tmp_df = tmp_df.iloc[np.random.choice(np.arange(34000), 17000).tolist()]
			tmp_df.to_csv('results/RR2results/' + file_name + '.csv', index = False)



'''
summarizing results at base case
'''


import numpy as np
import pandas as pd
import json
import seaborn as sns
import matplotlib as mpl
import matplotlib.gridspec as gridspec
print(mpl.rcParams['backend'])
mpl.use('TkAgg')
import matplotlib.pyplot as plt
plt.get_backend()
import copy

import os
os.chdir("/Users/szu-yukao/Documents/Network_structure_and_STI/networkSTI")
cwd = os.getcwd()
print(cwd)

txt = '(corr_scr 2yrs)'
graph_key = ['random', 'community', 'power_law', 'empirical']

for cor_or_not in ['Corr']: # "Uncorr"
	for lvl in ['Low', 'High']: 
		output_list = [None] * len(graph_key)
		for g in range(len(graph_key)): 
			mydf = pd.read_csv('results/RR2results/' + graph_key[g] + cor_or_not + lvl + txt + '.csv')
			mydf['pr'] = mydf['p_treat_pn'] + mydf['p_treat_ept'] + mydf['p_treat_tr']
			mydf['p_infPart6mon'] = mydf['n_infPart6mon'] / mydf['n_part6mon']
			mydf['averageTimesInfected_norm'] = mydf['pEverInfected'] * mydf['averageTimesInfected']
			mydf['testYield'] = mydf['nTest'] - mydf['nScreen']
			mydf['EPTYield'] = mydf['nTrueTreatIntervention'] + mydf['nOvertreat']
			mydf['pPartReachedTested'] = mydf['out_nPartnerTestedBefore'] / mydf['nIntervention']
			mydf['avgDur'] = mydf['tot_person_time'] / mydf['newI']
			sum_out = mydf.groupby(['strategy', 'pr']).agg([np.mean, np.std])
			sum_out = sum_out.stack()
			# sum_out[['I', 'newI', 'tot_person_time', 'TotalCost']].to_csv('results/RRsummary/' + 'sum_' + graph + cor_or_not + lvl + txt + '.csv')
			ix_ls = [('screen', 0, 'mean'), ('screen', 0, 'std'), ('pn', 0.71, 'mean'), ('pn', 0.71, 'std'), 
			('ept', 0.79, 'mean'), ('ept', 0.79, 'std'), ('trace', 0.79, 'mean'), ('trace', 0.79, 'std')]
			sum_out = sum_out.loc[ix_ls]
			sum_out['graph'] = graph_key[g]
			sum_out = sum_out[['graph', 'tot_person_time', 'TotalCost', 'I', 'newI', \
			'averageTimesInfected_norm', 'corr_timesInf_and_degree', \
			'n_part6mon', 'n_infPart6mon', 'p_infPart6mon', 'nIntervention', 'testYield', \
			'EPTYield', 'out_nPartnerTestedBefore', 'nTrueTreatIntervention', \
			'nDeliver', 'nContactTrace', 'nNotified', 'CostMedicine', 'CostTracing', 'CostTest', \
			'nExtSeed', 'nTest', 'nScreen', 'nTrueTreat', 'nOvertreat', 'nTrueTreatIntervention', \
			'pEfficient', 'pEverInfected', 'pEverBeenIntervention', 'avgTimesBeenIntervene', \
			'corr_PT_and_degree', 'pPartReachedTested', 'avgDur']]
		
			output_list[g] = sum_out
	
		output_list = pd.concat(output_list)
		output_list.to_csv('results/RR2summary/sum_' + cor_or_not + lvl + txt + '.csv')


randomDF = pd.read_csv('results/RR2results/randomCorrHigh(corr_scr 2yrs).csv')
randomDF = randomDF.loc[randomDF["strategy"] == "screen"]
communityDF = pd.read_csv('results/RR2results/communityCorrHigh(corr_scr 2yrs).csv')
communityDF = communityDF.loc[communityDF["strategy"] == "screen"]
power_lawDF = pd.read_csv('results/RR2results/power_lawCorrHigh(corr_scr 2yrs).csv')
power_lawDF = power_lawDF.loc[power_lawDF["strategy"] == "screen"]
empiricalDF = pd.read_csv('results/RR2results/empiricalCorrHigh(corr_scr 2yrs).csv')
empiricalDF = empiricalDF.loc[empiricalDF["strategy"] == "screen"]


fig = plt.figure(figsize = (12, 8))
G = gridspec.GridSpec(2, 2)
ax1 = plt.subplot(G[0, 0])
ax1.set_title('Random',fontsize=16)
ax1.hist(randomDF[["I"]].to_numpy().T[0], normed=True, bins=20, color = sns.xkcd_rgb['dodger blue'])

ax2 = plt.subplot(G[0, 1])
ax2.set_title('Community-structured',fontsize=16)
ax2.hist(communityDF[["I"]].to_numpy().T[0], normed=True, bins=20, color = sns.xkcd_rgb['dodger blue'])

ax3 = plt.subplot(G[1, 0])
ax3.set_title('Scale-free',fontsize=16)
ax3.hist(power_lawDF[["I"]].to_numpy().T[0], normed=True, bins=20, color = sns.xkcd_rgb['dodger blue'])

ax4 = plt.subplot(G[1, 1])
ax4.set_title('Empirical',fontsize=16)
ax4.hist(empiricalDF[["I"]].to_numpy().T[0], normed=True, bins=20, color = sns.xkcd_rgb['dodger blue'])

G.tight_layout(fig, rect=[0, 0, 1, 0.97])
plt.savefig('results/RR2summary/prevalence distribution (high).png', format='png', dpi=600)
plt.clf()



'''
managing network summary statistics from network simulation
'''

import numpy as np
import netSTI.netSTI as netSTI
import pandas as pd
import json
import matplotlib as mpl
from sklearn.linear_model import LinearRegression
import copy

import os
os.chdir("/Users/szu-yukao/Documents/Network_structure_and_STI/networkSTI")
cwd = os.getcwd()
print(cwd)

graph_ls = ['random', 'community', 'power_law', 'empirical']


for graph in graph_ls: 
	temp_ls = []
	for file_n in range(1, 11): 
		with open('results/netout/netout_'+ graph + '_Corr' + '_' + str(file_n)+'.txt') as json_file:  
			temp_ls += json.load(json_file)

	with open('results/netout/netout_'+ graph + 'Corr' + '1000sim.txt', 'w') as json_file:  
		json.dump(temp_ls, json_file)










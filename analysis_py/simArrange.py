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

	for file_n in range(1, 11): 
		try: 
			with open('results/RRresults/'+ graph + cor_or_not + lvl + '_' + str(file_n)+'.txt') as json_file:  
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


for cor_or_not in ['Uncorr', 'Corr']:
	for lvl in ['Low', 'High']: 
		for graph in graph_key: 
			tmp_df, file_name = get_sim_result(graph, lvl, cor_or_not, texts = txt)
			tmp_df.to_csv('results/RRresults/' + file_name + '.csv', index = False)



'''
summarizing results at base case
'''


import numpy as np
import pandas as pd
import json
import matplotlib as mpl
import copy

import os
os.chdir("/Users/szu-yukao/Documents/Network_structure_and_STI/networkSTI")
cwd = os.getcwd()
print(cwd)

txt = '(corr_scr 2yrs)'
graph_key = ['random', 'community', 'power_law', 'empirical']

for cor_or_not in ['Uncorr', 'Corr']:
	for lvl in ['Low', 'High']: 
		output_list = [None] * len(graph_key)
		for g in range(len(graph_key)): 
			mydf = pd.read_csv('results/RRresults/' + graph_key[g] + cor_or_not + lvl + txt + '.csv')
			mydf['pr'] = mydf['p_treat_pn'] + mydf['p_treat_ept'] + mydf['p_treat_tr']
			mydf['averageTimesInfected_norm'] = mydf['pEverInfected'] * mydf['averageTimesInfected']
			sum_out = mydf.groupby(['strategy', 'pr']).agg([np.mean, np.std])
			sum_out = sum_out.stack()
			# sum_out[['I', 'newI', 'tot_person_time', 'TotalCost']].to_csv('results/RRsummary/' + 'sum_' + graph + cor_or_not + lvl + txt + '.csv')
			ix_ls = [('screen', 0, 'mean'), ('screen', 0, 'std'), ('pn', 0.71, 'mean'), ('pn', 0.71, 'std'), 
			('ept', 0.79, 'mean'), ('ept', 0.79, 'std'), ('trace', 0.79, 'mean'), ('trace', 0.79, 'std')]
			sum_out = sum_out.loc[ix_ls]
			sum_out['graph'] = graph_key[g]
			sum_out = sum_out[['graph', 'I', 'newI', 'averageTimesInfected_norm', 'corr_timesInf_and_degree', \
			'tot_person_time', 'TotalCost', 'CostMedicine', 'CostTracing', 'CostTest', 
			'nExtSeed', 'nTest', 'nScreen', 'nDeliver', 'nContactTrace', 'nNotified', 'nTrueTreat', 'nOvertreat', 
			'nTrueTreatIntervention', 'pEfficient', 'pEverInfected', \
			'nIntervention', 'pEverBeenIntervention', 'avgTimesBeenIntervene', 
			'corr_PT_and_degree']]
		
			output_list[g] = sum_out
	
		output_list = pd.concat(output_list)
		output_list.to_csv('results/RRsummary/sum_' + cor_or_not + lvl + txt + '.csv')



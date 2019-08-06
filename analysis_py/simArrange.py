import numpy as np
import netSTI.net as net
import pandas as pd
import json
import matplotlib as mpl
from sklearn.linear_model import LinearRegression
import copy


def get_prev_level(tar_prev): 
	if tar_prev == 0.2: 
		lvl = "High"
	else: 
		lvl = "Low"
	return lvl

def get_sim_result(graph, strategy, tar_prev, cor_or_not): 
	'''
	graph = 'random'
	strategy = 'PN'
	'''
	lvl = get_prev_level(tar_prev)

	if strategy != 'null': 
		var_param = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.71, 0.79, 0.8, 0.9, 1]

		all_data = [None] * (len(var_param))

		for v in range(1, 11): 
			p_treat_PN = 0
			p_treat_ept = 0
			p_treat_tr = 0

			if strategy == 'PN': 
				p_treat_PN = var_param[v]
			if strategy == 'EPT': 
				p_treat_ept = var_param[v]
			if strategy == 'tracing': 
				p_treat_tr = var_param[v]

			temp_ls = []

			for file_n in range(1, 11): 
				try: 
					with open('results/RRresults/'+ graph + lvl + cor_or_not + '_' + strategy + '_' + str(var_param[v]) + '_' + str(file_n)+'.txt') as json_file:  
						temp_ls += json.load(json_file)
				except: 
					print(graph + lvl + cor_or_not + '_' + strategy + '_' + str(var_param[v]) + '_' + str(file_n))

			if len(temp_ls) > 0: 
				key_ls = [[k for k, v in temp_ls[0].items()][0]] + [k for k, v in temp_ls[0]['summary'].items()]
				val = np.zeros((len(temp_ls), len(key_ls))).T.tolist()

				tmp_data = dict(zip(key_ls, val))

				for i in range(len(temp_ls)): 
					tmp_data['run'][i] = temp_ls[i]['run']
					for j in [k for k, v in temp_ls[0]['summary'].items()]: 
						tmp_data[j][i] = temp_ls[i]['summary'][j]

				tmp_data = pd.DataFrame(tmp_data)
				tmp_data['p_treat_PN'] = p_treat_PN
				tmp_data['p_treat_ept'] = p_treat_ept
				tmp_data['p_treat_tr'] = p_treat_tr

				all_data[v] = tmp_data

		out = pd.concat(all_data)
	else: 
		p_treat_PN = 0
		p_treat_ept = 0
		p_treat_tr = 0

		temp_ls = []

		for file_n in range(1, 11): 
			try: 
				with open('results/RRresults/'+ graph + lvl + cor_or_not + '_' + strategy + '_' + str(file_n)+'.txt') as json_file:  
					temp_ls += json.load(json_file)
			except: 
				print(graph + lvl + cor_or_not + '_' + strategy + '_' + str(file_n))

		if len(temp_ls) > 0: 
			key_ls = [[k for k, v in temp_ls[0].items()][0]] + [k for k, v in temp_ls[0]['summary'].items()]
			val = np.zeros((len(temp_ls), len(key_ls))).T.tolist()

			tmp_data = dict(zip(key_ls, val))

			for i in range(len(temp_ls)): 
				tmp_data['run'][i] = temp_ls[i]['run']
				for j in [k for k, v in temp_ls[0]['summary'].items()]: 
					tmp_data[j][i] = temp_ls[i]['summary'][j]

			tmp_data = pd.DataFrame(tmp_data)
			tmp_data['p_treat_PN'] = p_treat_PN
			tmp_data['p_treat_ept'] = p_treat_ept
			tmp_data['p_treat_tr'] = p_treat_tr

		out = tmp_data

	out['strategy'] = strategy

	file_name = graph + lvl + cor_or_not + '_' + strategy

	return out, file_name



graph_key = ['random', 'community', 'power_law']
strategy_key = ['null', 'PN', 'EPT', 'tracing']

tar_prev = 0.05
lvl = get_prev_level(tar_prev)
cor_or_not = "Uncorr"

for graph in graph_key: 
	for strategy in strategy_key: 
		tmp_df, file_name = get_sim_result(graph, strategy, tar_prev, cor_or_not)
		tmp_df.to_csv('results/RRresults/' + file_name + '.csv', index = False)



import numpy as np
import netSTI.net as net
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


# Basecase

tar_prev = 0.05
lvl = get_prev_level(tar_prev)
cor_or_not = "Uncorr"

graph_key = ['random', 'community', 'power_law']
strategy_key = ['null', 'PN', 'EPT', 'tracing']

p_treat_PN = 0.71
p_treat_ept = 0.79
p_treat_tr = 0.79


for graph in graph_key: 
	output = [None] * len(strategy_key)

	for i in range(len(strategy_key)): 

		file_name = graph + lvl + cor_or_not + '_' + strategy_key[i]

		df = pd.read_csv('results/RRresults/' + file_name + '.csv')
		tmp_df = df

		if strategy_key[i] == 'PN': 
			tmp_df = df.loc[df['p_treat_PN'] == p_treat_PN]
		if strategy_key[i] == 'EPT': 
			tmp_df = df.loc[df['p_treat_ept'] == p_treat_ept]
		if strategy_key[i] == 'tracing': 
			tmp_df = df.loc[df['p_treat_tr'] == p_treat_tr]
		
		tmp_df = tmp_df.describe().T[["mean", "std"]]
		tmp_df['strategy'] = strategy_key[i]
		tmp_df = tmp_df.T
		tmp_df['names'] = tmp_df.index
		tmp_df.names = pd.Categorical(tmp_df.names, categories=['strategy', 'mean', 'std'])
		tmp_df = tmp_df.sort_values('names')
		
		output[i] = tmp_df

	output = pd.concat(output)
	output = output[['names', 'I', 'S', 'newI', 'nDeliver', 'nContactTrace', 'nNotified', 
			'nTrueTreat', 'nOvertreat', 'nTest', 'nScreen', 'nTrueTreatIntervention', 'nIntervention', 
			'pEfficient', 'pEverInfected', 'averageTimesInfected', 
			'pEverBeenIntervention', 'avgTimesBeenIntervene', 
			'CostMedicine', 'CostTest', 'CostTracing', 'TotalCost', 'avg_degree',
			'p_treat_PN', 'p_treat_ept', 'p_treat_tr', 'run', 'strategy']]
	output.to_csv('results/RRresults/' + graph + lvl + cor_or_not + '_sumout.csv')




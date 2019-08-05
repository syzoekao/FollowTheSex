import timeit
import numpy as np
import json
import scipy.stats as stats
import os
os.chdir("/Users/szu-yukao/Documents/Network_structure_and_STI/networkSTI")
cwd = os.getcwd()
print(cwd)

import matplotlib as mpl
print(mpl.rcParams['backend'])
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
plt.get_backend()


for g in ["random", "community", "power_law"]: 
	for level in ["Low", "High"]: 
		for indep in ["Uncorr", "Corr"]: 
			
			ret = []
			for i in range(1, 11): 
				file_name = g + level + indep
				try:  
					with open('results/SIRresults/' + file_name + str(i) + '.txt') as json_file:  
						ret += json.load(json_file)
				except FileNotFoundError:
					print(file_name + str(i))

			with open('results/SIRresults/' + file_name + '.txt', 'w') as json_file: 
				json.dump(ret, json_file)


from matplotlib.patches import Rectangle

level = "High"
indep = "Corr"
graph = ["random", "community", "power_law"]
bar_col = ["cornflowerblue", "limegreen", "salmon"]
line_type = ['-', '--', ':']
lty = [None] * len(graph)
post_mean = [None] * len(graph)
post_sd = [None] * len(graph)

fig = plt.figure(figsize=(6,4))

for i in range(len(graph)):
	file_name = graph[i] + level + indep

	with open('results/SIRresults/' + file_name + '.txt') as json_file:  
		ret = np.array(json.load(json_file))
		
	par = ret[:, 0]
	weight = ret[:, 3] / np.sum(ret[:, 3])

	density, bins = np.histogram(par, density = True, weights = weight, bins = 50)
	unity_density = density / density.sum()
	widths = bins[:-1] - bins[1:]
	w_mean = np.average(par, weights = weight)
	w_sd = np.sqrt(np.average((par - w_mean)**2, weights=weight))

	plt.bar(bins[1:], unity_density, width = widths, color = bar_col[i], alpha = 0.75)
	lty[i] = plt.axvline(np.sum(par * weight), color = 'k', linestyle = line_type[i])

	post_mean[i] = str(np.round(w_mean, 1))
	post_sd[i] = str(np.round(w_sd, 2))

plt.text(110, 0.03, 'random: $\\mu$=' + post_mean[0] + '; $\\sigma$=' + post_sd[0] + \
	'\ncommunity: $\\mu$=' + post_mean[1] + '; $\\sigma$=' + post_sd[1] + \
	'\npower_law: $\\mu$=' + post_mean[2] + '; $\\sigma$=' + post_sd[2] , fontsize = 'x-small')

#create legend
handles = [Rectangle((0, 0), 1, 1, color = c, ec = c) for c in bar_col] + lty
labels = [graph[i] for i in range(len(graph))] + [graph[i] + ' (mean)'  for i in range(len(graph))]
plt.legend(handles, labels, loc = 1, frameon = False, fontsize = 'x-small')

plt.xlabel('number', fontsize = 14)
plt.ylabel('density', fontsize = 14)
plt.title('Posterior distribution of # of sex acts\nper partner per year', fontsize = 16)
plt.tight_layout()
plt.savefig('results/calibration_' + level + '_' + indep + '.png', dpi = 400)
plt.clf()


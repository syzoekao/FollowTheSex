'''
Check degree distribution of power law
'''
import numpy as np
import netSTI.netSTI as netSTI
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
years = 5
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
    rel_hist = netSTI.SIR_net_generator(run, Npop, years = 5, days = 14, 
    graph = "power_law", pEFoI = 0.0001, independent = True, base_case = False, 
    trend = False, analysis_window = 2, output_netsum = False, output_net = True)

    tmp_rel = rel_hist[(rel_hist[:, 3] >= (3 * unit_per_year))]
    tmp_names, tmp_degree = np.unique(tmp_rel[:, :2], return_counts = True)
    degree, counts = np.unique(tmp_degree, return_counts = True)
    degree_dist = counts / np.sum(counts)
    fit = np.polyfit(np.log(degree), np.log(degree_dist), 1)
    fit_vec[run] = fit[1]
    fit_fn = np.poly1d(fit)
    avg_deg[run] = np.sum(tmp_degree) / Npop
    ax.scatter(np.log(degree), np.log(degree_dist), edgecolor = 'limegreen', facecolors='none')
    ax.plot(np.log(degree), fit_fn(np.log(degree)), 'royalblue')

x_ticks = np.array([0.1, 1, 2, 3, 4, 5])
y_ticks = np.array([-8, -6, -4, -2, -0.1])
ax.set_xticks(x_ticks)
ax.set_xticklabels(np.round(np.exp(x_ticks)))
ax.set_yticks(y_ticks)
ax.set_yticklabels(np.round(np.exp(y_ticks), 3))
ax.text(0.1, -7, 'mean of the slope is\n' + str(np.round(np.mean(fit_vec), 2)) + \
    '\naverage mean degree\n(last 2 years):\n' + str(np.round(np.mean(avg_deg), 2)), fontsize=12)
plt.xlabel('degree')
plt.ylabel('proportion')
plt.title('100 network samples')
plt.savefig('results/power law degree distribution.eps', format='eps', dpi=1000)



'''
Network configuration plots
'''

import numpy as np
import timeit
import netSTI.netSTI as netSTI
import plotly.offline as offline
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.io as pio
import plotly.tools as tls
import cmocean
import networkx as nx
import pandas as pd
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib as mpl
print(mpl.rcParams['backend'])
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
plt.get_backend()
import itertools
import os
os.chdir("/Users/szu-yukao/Documents/Network_structure_and_STI/networkSTI")
cwd = os.getcwd()
print(cwd)

def cmocean_to_plotly(cmap, pl_entries):
	h = 1.0/(pl_entries-1)
	pl_colorscale = []
	for k in range(pl_entries):
		C = list(map(np.uint8, np.array(cmap(k*h)[:3])*255))
		pl_colorscale.append([k*h, 'rgb'+str((C[0], C[1], C[2]))])
	return pl_colorscale

graph = "empirical"
indep = False
n_yrs = 5
Npop = 1000
ID = np.arange(Npop)
 
el = netSTI.SIR_net_generator(500, Npop, years = 5, days = 14, 
    graph = graph, pEFoI = (1 / 5000) * 2, independent = indep, corr_scr = True, base_case = False, 
    trend = False, analysis_window = 5, output_netsum = False, output_net = True)


fig_window = (n_yrs * 26 - 4 * 26)
tmp_el = el[np.where(el[:, 3] >= fig_window)]

sex_array = np.zeros((Npop, Npop))
for i in range(tmp_el.shape[0]): 
	x = tmp_el[i, 0]
	y = tmp_el[i, 1]
	sex_array[x, y] = 1
	sex_array[y, x] = 1

G = nx.from_numpy_matrix(sex_array)
n_degree = np.sum(sex_array, axis = 1)
degree_dict = dict(zip(ID, n_degree))
nx.set_node_attributes(G, degree_dict, 'degree')

thermal = cmocean_to_plotly(cmocean.cm.thermal, n_degree.shape[0])

if graph != "power_law": 
	pos = graphviz_layout(G)
else: 
	pos = nx.spring_layout(G, k = 0.25)
nx.set_node_attributes(G, pos, 'pos') 

edge_trace = go.Scatter(
	x=[],
	y=[],
	line = dict(width = 0.5, color = '#888'),
	hoverinfo = 'none',
	mode = 'lines')

attr_pos = np.array([v for k, v in nx.get_node_attributes(G, 'pos').items()])
G_edge = np.array(G.edges())

edge_x_pos = np.vstack((attr_pos[G_edge[:, 0]][:, 0], attr_pos[G_edge[:, 1]][:, 0])).T.flatten()
edge_y_pos = np.vstack((attr_pos[G_edge[:, 0]][:, 1], attr_pos[G_edge[:, 1]][:, 1])).T.flatten()

edge_trace['x'] = edge_x_pos
edge_trace['y'] = edge_y_pos


node_trace = go.Scatter(
	x = [],
	y = [],
	text = [],
	mode = 'markers',
	opacity = 1,
	hoverinfo = 'text',
	marker = dict(
		showscale = True,
		colorscale = thermal, 
		reversescale = False,
		cmin = 0, 
		cmax = 50, 
		color = [],
		size = [],
		# colorbar = dict(
		# 	thickness = 15,
		# 	title = 'degree',
		# 	tickvals = np.arange(50), 
		# 	ticktext = np.arange(50), 
		# 	xanchor = 'left',
		# 	titleside = 'right'
		# ),
		line = dict(width = 2, 
				color = '#000')))

node_trace['x'] = attr_pos[:, 0]
node_trace['y'] = attr_pos[:, 1]

node_trace['marker']['size'] = np.repeat(10, Npop)
node_trace['marker']['color'] = n_degree

g_title = graph
if graph == "power_law": 
	g_title = "scale-free"

fig = go.Figure(data=[edge_trace, node_trace],
	layout = go.Layout(
		title = '<br>' + '(D) ' + g_title + ' network',
		titlefont = dict(size = 60),
		showlegend = False,
		hovermode = 'closest',
		margin = dict(b = 20, l = 5, r = 5, t = 40),
		xaxis = dict(showgrid=False, zeroline=False, showticklabels=False),
		yaxis = dict(showgrid=False, zeroline=False, showticklabels=False)))

pio.write_image(fig, file='results/netout/' + graph + '.png', format='png', 
	width=1200, height=1000, scale=4)


'''
Check network generation:  
average degree, correlation between degree and duration 
'''

import numpy as np
import timeit
import json
import netSTI.netSTI as netSTI
import plotly.offline as offline
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.io as pio
import plotly.tools as tls
import cmocean
import networkx as nx
import pandas as pd
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib as mpl
print(mpl.rcParams['backend'])
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
plt.get_backend()
import itertools
import os
os.chdir("/Users/szu-yukao/Documents/Network_structure_and_STI/networkSTI")
cwd = os.getcwd()
print(cwd)


from matplotlib.patches import Rectangle

cor_or_not = 'Corr'
graph = ["power_law", "empirical", "random", "community"]
bar_col = ["salmon", "gold", "cornflowerblue", "limegreen"]
line_type = [':', '-.', '-', '--']
lty = [None] * len(graph)
post_mean = [None] * len(graph)
post_sd = [None] * len(graph)

plt.close('all')

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows = 2, ncols = 3, figsize=(16, 8))
for g in range(len(graph)): 
	with open('results/netout/netout_' + graph[g] + '' + cor_or_not + '1000sim.txt') as json_file:  
		ret = json.load(json_file)

	key_ls = [k for k, v in ret[0].items()]
	result = np.zeros((len(ret), len(key_ls)))
	for i in range(len(ret)):
		result[i] = [v for k, v in ret[i].items()] 

	density, bins = np.histogram(result[:, 0], normed = True, density = True, bins = 20)
	unity_density = density / density.sum()
	widths = bins[:-1] - bins[1:]
	ax1.bar(bins[1:], unity_density, width = widths, color = bar_col[g], alpha = 0.6)
	lty[g] = ax1.axvline(result[:, 0].mean(), color = 'k', linestyle = line_type[g])
	# ax1.text(result[:, 0].mean(), 0, str(np.round(result[:, 0].mean(), 3)))
	ax1.set_xlim([17, 22])
	ax1.set_xlabel('degree')
	ax1.set_ylabel('density')
	ax1.set_title('Degree over 5 years')

	density, bins = np.histogram(result[:, 1], normed = True, density = True, bins = 20)
	unity_density = density / density.sum()
	widths = bins[:-1] - bins[1:]
	ax2.bar(bins[1:], unity_density, width = widths, color = bar_col[g], alpha = 0.6)
	lty[g] = ax2.axvline(result[:, 1].mean(), color = 'k', linestyle = line_type[g])
	# ax2.text(result[:, 1].mean(), 0, str(np.round(result[:, 1].mean(), 3)))
	ax2.set_xlim([7, 10])
	ax2.set_xlabel('degree')
	ax2.set_ylabel('density')
	ax2.set_title('Degree over the evaluation window\n(last 2 years)')

	density, bins = np.histogram(result[:, 2], normed = True, density = True, bins = 20)
	unity_density = density / density.sum()
	widths = bins[:-1] - bins[1:]
	ax3.bar(bins[1:], unity_density, width = widths, color = bar_col[g], alpha = 0.6)
	lty[g] = ax3.axvline(result[:, 2].mean(), color = 'k', linestyle = line_type[g])
	# ax5.text(result[:, 2].mean(), 0, str(np.round(result[:, 2].mean(), 3)))
	ax3.set_xlim([1, 1.4])
	ax3.set_xlabel('degree')
	ax3.set_ylabel('density')
	ax3.set_title('Degree at each time step')

	density, bins = np.histogram(result[:, 3], normed = True, density = True, bins = 20)
	unity_density = density / density.sum()
	widths = bins[:-1] - bins[1:]
	ax4.bar(bins[1:], unity_density, width = widths, color = bar_col[g], alpha = 0.6)
	lty[g] = ax4.axvline(result[:, 3].mean(), color = 'k', linestyle = line_type[g])
	# ax4.text(result[:, 3].mean(), 0, str(np.round(result[:, 3].mean(), 3)))
	ax4.set_xlim([0.2, 0.65])
	ax4.set_xlabel('proportion')
	ax4.set_ylabel('density')
	ax4.set_title('Proportion individuals who have\nno partners at each time step')

	density, bins = np.histogram(result[:, 4], normed = True, density = True, bins = 20)
	unity_density = density / density.sum()
	widths = bins[:-1] - bins[1:]
	ax5.bar(bins[1:], unity_density, width = widths, color = bar_col[g], alpha = 0.6)
	lty[g] = ax5.axvline(result[:, 4].mean(), color = 'k', linestyle = line_type[g])
	# ax3.text(result[:, 4].mean(), 0, str(np.round(result[:, 4].mean(), 3)))
	ax5.set_xlim([-0.13, -0.06])
	ax5.set_xlabel('correlation')
	ax5.set_ylabel('density')
	ax5.set_title('Correlation between duration and\nsum of degrees of a pair of individuals')

	density, bins = np.histogram(result[:, 5], normed = True, density = True, bins = 20)
	unity_density = density / density.sum()
	widths = bins[:-1] - bins[1:]
	ax6.bar(bins[1:], unity_density, width = widths, color = bar_col[g], alpha = 0.6)
	lty[g] = ax6.axvline((result[:, 5].mean()), color = 'k', linestyle = line_type[g])
	# ax3.text(result[:, 5].mean(), 0, str(np.round(result[:, 5].mean(), 3)))
	ax6.set_xlim([3.75, 4.75])
	ax6.set_xlabel('months')
	ax6.set_ylabel('density')
	ax6.set_title('Average duration')


# fig.subplots_adjust(bottom = 0.3) 
handles = [Rectangle((0, 0), 1, 1, color = c, ec = c) for c in bar_col] + lty
labels = ["scale-free", "empirical", "random", "community"] + \
["scale-free\n(mean)", "empirical\n(mean)", "random\n(mean)", "community\n(mean)"]
plt.legend(handles, labels, ncol = 1, bbox_to_anchor = (1.35, 0), loc='lower right', 
	borderaxespad = 0, frameon = False, fontsize = 'medium')

plt.tight_layout()
plt.savefig('results/netout/netout_' + cor_or_not + '1000sim.png', format='png', dpi=500)
plt.clf()





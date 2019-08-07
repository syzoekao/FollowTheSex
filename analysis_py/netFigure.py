import numpy as np
import timeit
import netSTI.net as net
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
 
el = net.SIR_net_generator(run = 30, Npop = Npop, years = 5, days = 14, 
    graph = graph, pEFoI = (1 / 5000) / 2 , 
    pContact_PN = 0.49, pContact_ept = 0.7, pContact_tr = 0.7, 
    p_treat_PN = 0.71, p_treat_ept = 0.79, p_treat_tr = 0.79, 
    independent = indep, calibration = False, 
    analysis_window = 2, output_netsum = False, output_net = True)


fig_window = (n_yrs * 26 - 3 * 26)
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
		cmax = 41, 
		color = [],
		size = [],
		colorbar = dict(
			thickness = 15,
			title = 'degree',
			tickvals = np.arange(41), 
			ticktext = np.arange(41), 
			xanchor = 'left',
			titleside = 'right'
		),
		line = dict(width = 2, 
				color = '#000')))

node_trace['x'] = attr_pos[:, 0]
node_trace['y'] = attr_pos[:, 1]

node_trace['marker']['size'] = np.repeat(10, Npop)
node_trace['marker']['color'] = n_degree

fig = go.Figure(data=[edge_trace, node_trace],
	layout = go.Layout(
		title = '<br>' + graph + ' network',
		titlefont = dict(size = 30),
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
import netSTI.net as net
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
	with open('results/netout/netout_' + graph[g] + '_' + cor_or_not + '.txt') as json_file:  
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
	ax1.set_xlabel('degree')
	ax1.set_ylabel('density')
	ax1.set_title('Degree over 5 years')

	density, bins = np.histogram(result[:, 1], normed = True, density = True, bins = 20)
	unity_density = density / density.sum()
	widths = bins[:-1] - bins[1:]
	ax2.bar(bins[1:], unity_density, width = widths, color = bar_col[g], alpha = 0.6)
	lty[g] = ax2.axvline(result[:, 1].mean(), color = 'k', linestyle = line_type[g])
	# ax2.text(result[:, 1].mean(), 0, str(np.round(result[:, 1].mean(), 3)))
	ax2.set_xlabel('degree')
	ax2.set_ylabel('density')
	ax2.set_title('Degree over the evaluation window')

	density, bins = np.histogram(result[:, 2], normed = True, density = True, bins = 20)
	unity_density = density / density.sum()
	widths = bins[:-1] - bins[1:]
	ax3.bar(bins[1:], unity_density, width = widths, color = bar_col[g], alpha = 0.6)
	lty[g] = ax3.axvline(result[:, 2].mean(), color = 'k', linestyle = line_type[g])
	# ax5.text(result[:, 2].mean(), 0, str(np.round(result[:, 2].mean(), 3)))
	ax3.set_xlabel('degree')
	ax3.set_ylabel('density')
	ax3.set_title('Degree at each time step')

	density, bins = np.histogram(result[:, 3], normed = True, density = True, bins = 20)
	unity_density = density / density.sum()
	widths = bins[:-1] - bins[1:]
	ax4.bar(bins[1:], unity_density, width = widths, color = bar_col[g], alpha = 0.6)
	lty[g] = ax4.axvline(result[:, 3].mean(), color = 'k', linestyle = line_type[g])
	# ax4.text(result[:, 3].mean(), 0, str(np.round(result[:, 3].mean(), 3)))
	ax4.set_xlabel('proportion')
	ax4.set_ylabel('density')
	ax4.set_title('proportion individuals who have no partners\nover evaluation window')

	density, bins = np.histogram(result[:, 4], normed = True, density = True, bins = 20)
	unity_density = density / density.sum()
	widths = bins[:-1] - bins[1:]
	ax5.bar(bins[1:], unity_density, width = widths, color = bar_col[g], alpha = 0.6)
	lty[g] = ax5.axvline(result[:, 4].mean(), color = 'k', linestyle = line_type[g])
	# ax3.text(result[:, 4].mean(), 0, str(np.round(result[:, 4].mean(), 3)))
	ax5.set_xlabel('correlation')
	ax5.set_ylabel('density')
	ax5.set_title('Correlation between degree and duration')

	density, bins = np.histogram(result[:, 5]/2, normed = True, density = True, bins = 20)
	unity_density = density / density.sum()
	widths = bins[:-1] - bins[1:]
	ax6.bar(bins[1:], unity_density, width = widths, color = bar_col[g], alpha = 0.6)
	lty[g] = ax6.axvline((result[:, 5].mean()) / 2, color = 'k', linestyle = line_type[g])
	# ax3.text(result[:, 5].mean(), 0, str(np.round(result[:, 5].mean(), 3)))
	ax6.set_xlabel('months')
	ax6.set_ylabel('density')
	ax6.set_title('average duration')


# fig.subplots_adjust(bottom = 0.3) 
handles = [Rectangle((0, 0), 1, 1, color = c, ec = c) for c in bar_col] + lty
labels = [graph[i] for i in range(len(graph))] + [graph[i] + '\n(mean)'  for i in range(len(graph))]
plt.legend(handles, labels, ncol = 1, bbox_to_anchor = (1.35, 0), loc='lower right', 
	borderaxespad = 0, frameon = False, fontsize = 'medium')

plt.tight_layout()
plt.savefig('results/netout/netout_' + cor_or_not + '.png', format='png', dpi=500)
plt.clf()





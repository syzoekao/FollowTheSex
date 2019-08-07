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

graph = "power_law"
independent = False
n_yrs = 50
Npop = 1000
ID = np.arange(Npop)
 
el = net.SIR_net_generator(24, 1, Npop, years = n_yrs, days = 14, 
	strategy = "null", graph = graph, independent = independent, \
	calibration = False, analysis_window = 20, output_net = True)

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

pio.write_image(fig, file='results/' + graph + '.png', format='png', 
	width=1200, height=1000, scale=4)


'''
Check network generation:  
average degree, correlation between degree and duration 
'''

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

years = 5
days = 14
unit_per_year = int(365 / days)
time_horizon = unit_per_year * years
Npop = 5000

n_sim = 500
cor_vec = np.zeros(n_sim)
mean_deg = np.zeros(n_sim)
mean_iso = np.zeros(n_sim)
for z in range(n_sim): 
	print(z)
	run = np.random.choice(np.arange(100000), 1)
	g, avg_deg = net.SIR_net_generator(run, Npop = Npop, years = 5, days = 14, 
		graph = "random", pEFoI = (1 / 5000) / 2 , 
		pContact_PN = 0.49, pContact_ept = 0.7, pContact_tr = 0.7, 
		p_treat_PN = 0.71, p_treat_ept = 0.79, p_treat_tr = 0.79, 
		independent = False, calibration = False, 
		analysis_window = 2, output_net = True)
	
	tmp_ix, tmp_degree = np.unique(g[:, :2], return_counts = True)
	mean_deg[z] = np.sum(tmp_degree) / Npop

	tmpDur = np.zeros((Npop, Npop))
	tmpDur[:, :] = np.nan
	g_dur = g[:, 3] - g[:, 2]
	for i in range(g.shape[0]): 
		x = g[i, 0]
		y = g[i, 1]
		tmpDur[x, y] = g_dur[i]
		tmpDur[y, x] = g_dur[i]
	tmp_dur = np.nanmean(tmpDur, axis = 1)
	cor_vec[z] = np.round(np.corrcoef(tmp_degree, tmp_dur)[0, 1], 3)

	# check isolates in the last 10 years
	t_beg = time_horizon - unit_per_year * 2
	t_end = time_horizon
	iso_vec = np.zeros(t_end - t_beg)
	for t in range(t_beg, t_end): 
		tmp_g = g[np.where((g[:, 2] <= t) & (g[:, 3] > t))]
		in_rel = np.unique(tmp_g[:, :2])
		iso_vec[t - t_beg] = 1 - in_rel.shape[0] / Npop
	mean_iso[z] = np.mean(iso_vec)

fig, (ax1, ax2, ax3) = plt.subplots(nrows = 1, ncols = 3, figsize=(18,4))
density, bins = np.histogram(mean_deg, normed = True, density = True, bins = 20)
unity_density = density / density.sum()
widths = bins[:-1] - bins[1:]
ax1.bar(bins[1:], unity_density, width = widths, color = "cornflowerblue")
ax1.axvline(mean_deg.mean(), color = 'orangered')
ax1.text(mean_deg.mean(), 0, str(np.round(mean_deg.mean(), 3)))
ax1.set_xlabel('degree')
ax1.set_ylabel('density')
ax1.set_title('average degree\n(' + str(n_sim) + 'simulations)')

density, bins = np.histogram(cor_vec, normed = True, density = True, bins = 20)
unity_density = density / density.sum()
widths = bins[:-1] - bins[1:]
ax2.bar(bins[1:], unity_density, width = widths, color = "cornflowerblue")
ax2.axvline(cor_vec.mean(), color = 'orangered')
ax2.text(cor_vec.mean(), 0, str(np.round(cor_vec.mean(), 3)))
ax2.set_xlabel('correlation coefficient')
ax2.set_ylabel('density')
ax2.set_title('correlation coefficient between degree and duration\n(' + str(n_sim) + 'simulations)')

density, bins = np.histogram(mean_iso, normed = True, density = True, bins = 20)
unity_density = density / density.sum()
widths = bins[:-1] - bins[1:]
ax3.bar(bins[1:], unity_density, width = widths, color = "cornflowerblue")
ax3.axvline(mean_iso.mean(), color = 'orangered')
ax3.text(mean_iso.mean(), 0, str(np.round(mean_iso.mean(), 3)))
ax3.set_xlabel('%')
ax3.set_ylabel('density')
ax3.set_title('% population has no partners \n(' + str(n_sim) + 'simulations)')

plt.tight_layout()
plt.savefig('results/random dd cc duration (correlated).eps', format='eps', dpi=500)
plt.clf()
	

n_sim = 500
cor_vec = np.zeros(n_sim)
p_in_vec = np.zeros(n_sim)
mean_deg = np.zeros(n_sim)
mean_iso = np.zeros(n_sim)

same_cluster = np.tile(ID_cluster, (Npop, 1)).T == np.tile(ID_cluster, (Npop, 1))

for z in range(n_sim): 
	print(z)
	run = np.random.choice(np.arange(100000), 1)
	g, avg_deg = net.SIR_net_generator(run, Npop = Npop, years = 5, days = 14, 
		graph = "community", pEFoI = (1 / 5000) / 2 , 
		pContact_PN = 0.49, pContact_ept = 0.7, pContact_tr = 0.7, 
		p_treat_PN = 0.71, p_treat_ept = 0.79, p_treat_tr = 0.79, 
		independent = False, calibration = False, 
		analysis_window = 2, output_net = True)

	tmp_ix, tmp_degree = np.unique(g[:, :2], return_counts = True)
	mean_deg[z] = np.sum(tmp_degree) / Npop

	tmpDur = np.zeros((Npop, Npop))
	tmpDur[:, :] = np.nan
	g_dur = g[:, 3] - g[:, 2]
	for i in range(g.shape[0]): 
		x = g[i, 0]
		y = g[i, 1]
		tmpDur[x, y] = g_dur[i]
		tmpDur[y, x] = g_dur[i]
	tmp_dur = np.nanmean(tmpDur, axis = 1)
	cor_vec[z] = np.round(np.corrcoef(tmp_degree, tmp_dur)[0, 1], 3)

	tmpCluster = tmpMatrix * same_cluster
	tmp_in_degree = np.sum(tmpCluster, axis = 1)
	p_in_vec[z] = np.mean(tmp_in_degree / tmp_degree)

	# check isolates in the last 10 years
	t_beg = time_horizon - 26 * 2
	t_end = time_horizon
	iso_vec = np.zeros(t_end - t_beg)
	for t in range(t_beg, t_end): 
		tmp_g = g[np.where((g[:, 2] <= t) & (g[:, 3] > t))]
		in_rel = np.unique(tmp_g[:, :2])
		iso_vec[t - t_beg] = 1 - in_rel.shape[0] / Npop
	mean_iso[z] = np.mean(iso_vec)


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows = 2, ncols = 2, figsize=(12, 8))
density, bins = np.histogram(mean_deg, normed = True, density = True, bins = 20)
unity_density = density / density.sum()
widths = bins[:-1] - bins[1:]
ax1.bar(bins[1:], unity_density, width = widths, color = "cornflowerblue")
ax1.axvline(mean_deg.mean(), color = 'orangered')
ax1.text(mean_deg.mean(), 0, str(np.round(mean_deg.mean(), 3)))
ax1.set_xlabel('degree')
ax1.set_ylabel('density')
ax1.set_title('average degree\n(' + str(n_sim) + 'simulations)')

density, bins = np.histogram(cor_vec, normed = True, density = True, bins = 20)
unity_density = density / density.sum()
widths = bins[:-1] - bins[1:]
ax2.bar(bins[1:], unity_density, width = widths, color = "cornflowerblue")
ax2.axvline(cor_vec.mean(), color = 'orangered')
ax2.text(cor_vec.mean(), 0, str(np.round(cor_vec.mean(), 3)))
ax2.set_xlabel('correlation coefficient')
ax2.set_ylabel('density')
ax2.set_title('correlation coefficient between degree and duration\n(' + str(n_sim) + 'simulations)')

density, bins = np.histogram(p_in_vec, normed = True, density = True, bins = 20)
unity_density = density / density.sum()
widths = bins[:-1] - bins[1:]
ax3.bar(bins[1:], unity_density, width = widths, color = "cornflowerblue")
ax3.axvline(p_in_vec.mean(), color = 'orangered')
ax3.text(p_in_vec.mean(), 0, str(np.round(p_in_vec.mean(), 3)))
ax3.set_xlabel('% partners')
ax3.set_ylabel('density')
ax3.set_title('% partners in the same community\n(' + str(n_sim) + 'simulations)')

density, bins = np.histogram(mean_iso, normed = True, density = True, bins = 20)
unity_density = density / density.sum()
widths = bins[:-1] - bins[1:]
ax4.bar(bins[1:], unity_density, width = widths, color = "cornflowerblue")
ax4.axvline(mean_iso.mean(), color = 'orangered')
ax4.text(mean_iso.mean(), 0, str(np.round(mean_iso.mean(), 3)))
ax4.set_xlabel('%')
ax4.set_ylabel('density')
ax4.set_title('% population has no partners \n(' + str(n_sim) + 'simulations)')

plt.tight_layout()
plt.savefig('results/community dd cc duration (correlated).eps', format='eps', dpi=500)
plt.clf()



n_sim = 500
cor_vec = np.zeros(n_sim)
mean_deg = np.zeros(n_sim)
mean_iso = np.zeros(n_sim)
for z in range(n_sim): 
	print(z)
	run = np.random.choice(np.arange(100000), 1)
	g, avg_deg = net.SIR_net_generator(run, Npop = Npop, years = 5, days = 14, 
		graph = "power_law", pEFoI = (1 / 5000) / 2 , 
		pContact_PN = 0.49, pContact_ept = 0.7, pContact_tr = 0.7, 
		p_treat_PN = 0.71, p_treat_ept = 0.79, p_treat_tr = 0.79, 
		independent = False, calibration = False, 
		analysis_window = 2, output_net = True)

	tmp_ix, tmp_n = np.unique(g[:, :2], return_counts = True)
	tmp_degree = np.zeros(Npop)
	tmp_degree[tmp_ix] = tmp_n
	mean_deg[z] = np.sum(tmp_degree) / Npop

	tmpDur = np.zeros((Npop, Npop))
	tmpDur[:, :] = np.nan
	g_dur = g[:, 3] - g[:, 2]
	for i in range(g.shape[0]): 
		x = g[i, 0]
		y = g[i, 1]
		tmpDur[x, y] = g_dur[i]
		tmpDur[y, x] = g_dur[i]
	tmp_dur = np.nanmean(tmpDur, axis = 1)
	tmp_ix = np.isnan(tmp_dur)
	cor_vec[z] = np.round(np.corrcoef(tmp_degree[~tmp_ix], tmp_dur[~tmp_ix])[0, 1], 3)

	# check isolates in the last 10 years
	t_beg = time_horizon - 26 * 2
	t_end = time_horizon
	iso_vec = np.zeros(t_end - t_beg)
	for t in range(t_beg, t_end): 
		tmp_g = g[np.where((g[:, 2] <= t) & (g[:, 3] > t))]
		in_rel = np.unique(tmp_g[:, :2])
		iso_vec[t - t_beg] = 1 - in_rel.shape[0] / Npop
	mean_iso[z] = np.mean(iso_vec)


fig, (ax1, ax2, ax3) = plt.subplots(nrows = 1, ncols = 3, figsize=(18, 4))
density, bins = np.histogram(mean_deg, normed = True, density = True, bins = 20)
unity_density = density / density.sum()
widths = bins[:-1] - bins[1:]
ax1.bar(bins[1:], unity_density, width = widths, color = "cornflowerblue")
ax1.axvline(mean_deg.mean(), color = 'orangered')
ax1.text(mean_deg.mean(), 0, str(np.round(mean_deg.mean(), 3)))
ax1.set_xlabel('degree')
ax1.set_ylabel('density')
ax1.set_title('average degree\n(' + str(n_sim) + 'simulations)')

density, bins = np.histogram(cor_vec, normed = True, density = True, bins = 20)
unity_density = density / density.sum()
widths = bins[:-1] - bins[1:]
ax2.bar(bins[1:], unity_density, width = widths, color = "cornflowerblue")
ax2.axvline(cor_vec.mean(), color = 'orangered')
ax2.text(cor_vec.mean(), 0, str(np.round(cor_vec.mean(), 3)))
ax2.set_xlabel('correlation coefficient')
ax2.set_ylabel('density')
ax2.set_title('correlation coefficient between degree and duration\n(' + str(n_sim) + 'simulations)')

density, bins = np.histogram(mean_iso, normed = True, density = True, bins = 20)
unity_density = density / density.sum()
widths = bins[:-1] - bins[1:]
ax3.bar(bins[1:], unity_density, width = widths, color = "cornflowerblue")
ax3.axvline(mean_iso.mean(), color = 'orangered')
ax3.text(mean_iso.mean(), 0, str(np.round(mean_iso.mean(), 3)))
ax3.set_xlabel('%')
ax3.set_ylabel('density')
ax3.set_title('% population has no partners \n(' + str(n_sim) + 'simulations)')

plt.tight_layout()
plt.savefig('results/power_law dd cc duration (correlated).eps', format='eps', dpi=500)
plt.clf()




n_sim = 500
cor_vec = np.zeros(n_sim)
mean_deg = np.zeros(n_sim)
mean_iso = np.zeros(n_sim)
for z in range(n_sim): 
	run = np.random.choice(np.arange(10000), 1)
	print(z)
	g, avg_deg = net.SIR_net_generator(run, Npop = Npop, years = 5, days = 14, 
		graph = "empirical", pEFoI = (1 / 5000) / 2 , 
		pContact_PN = 0.49, pContact_ept = 0.7, pContact_tr = 0.7, 
		p_treat_PN = 0.71, p_treat_ept = 0.79, p_treat_tr = 0.79, 
		independent = False, calibration = False, 
		analysis_window = 2, output_net = True)

	tmp_ix, tmp_n = np.unique(g[:, :2], return_counts = True)
	tmp_degree = np.zeros(Npop)
	tmp_degree[tmp_ix] = tmp_n
	mean_deg[z] = np.sum(tmp_degree) / Npop

	tmpDur = np.zeros((Npop, Npop))
	tmpDur[:, :] = np.nan
	g_dur = g[:, 3] - g[:, 2]
	for i in range(g.shape[0]): 
		x = g[i, 0]
		y = g[i, 1]
		tmpDur[x, y] = g_dur[i]
		tmpDur[y, x] = g_dur[i]
	tmp_dur = np.nanmean(tmpDur, axis = 1)
	tmp_ix = np.isnan(tmp_dur)
	cor_vec[z] = np.round(np.corrcoef(tmp_degree[~tmp_ix], tmp_dur[~tmp_ix])[0, 1], 3)

	# check isolates in the last 10 years
	t_beg = time_horizon - 26 * 10
	t_end = time_horizon
	iso_vec = np.zeros(t_end - t_beg)
	for t in range(t_beg, t_end): 
		tmp_g = g[np.where((g[:, 2] <= t) & (g[:, 3] > t))]
		in_rel = np.unique(tmp_g[:, :2])
		iso_vec[t - t_beg] = 1 - in_rel.shape[0] / Npop
	mean_iso[z] = np.mean(iso_vec)


fig, (ax1, ax2, ax3) = plt.subplots(nrows = 1, ncols = 3, figsize=(18, 4))
density, bins = np.histogram(mean_deg, normed = True, density = True, bins = 20)
unity_density = density / density.sum()
widths = bins[:-1] - bins[1:]
ax1.bar(bins[1:], unity_density, width = widths, color = "cornflowerblue")
ax1.axvline(mean_deg.mean(), color = 'orangered')
ax1.text(mean_deg.mean(), 0, str(np.round(mean_deg.mean(), 3)))
ax1.set_xlabel('degree')
ax1.set_ylabel('density')
ax1.set_title('average degree\n(' + str(n_sim) + 'simulations)')

density, bins = np.histogram(cor_vec, normed = True, density = True, bins = 20)
unity_density = density / density.sum()
widths = bins[:-1] - bins[1:]
ax2.bar(bins[1:], unity_density, width = widths, color = "cornflowerblue")
ax2.axvline(cor_vec.mean(), color = 'orangered')
ax2.text(cor_vec.mean(), 0, str(np.round(cor_vec.mean(), 3)))
ax2.set_xlabel('correlation coefficient')
ax2.set_ylabel('density')
ax2.set_title('correlation coefficient between degree and duration\n(' + str(n_sim) + 'simulations)')

density, bins = np.histogram(mean_iso, normed = True, density = True, bins = 20)
unity_density = density / density.sum()
widths = bins[:-1] - bins[1:]
ax3.bar(bins[1:], unity_density, width = widths, color = "cornflowerblue")
ax3.axvline(mean_iso.mean(), color = 'orangered')
ax3.text(mean_iso.mean(), 0, str(np.round(mean_iso.mean(), 3)))
ax3.set_xlabel('%')
ax3.set_ylabel('density')
ax3.set_title('% population has no partners \n(' + str(n_sim) + 'simulations)')

plt.tight_layout()
plt.savefig('results/empirical dd cc duration (correlated).eps', format='eps', dpi=500)
plt.clf()









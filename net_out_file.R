txt = function(i_file, graph, indep) {
  paste0("import timeit
import simplejson
import numpy as np
import netSTI.netSTI as netSTI
import json

import os
os.chdir(\"/panfs/roc/groups/0/ennse/kaoxx085/networkSTI\")
cwd = os.getcwd()
print(cwd)

file_n = ",i_file,"
n_runs = 200

graph = \"", graph, "\"
indep = ",indep,"
pEFoI = 0.001

if indep == True: 
\tcor_or_not = \"Uncorr\"
else: 
\tcor_or_not = \"Corr\"

if pEFoI == (1 / 5000) / 2: 
\tlvl = \"Low\"
else: 
\tlvl = \"High\"

result_l = [None] * n_runs

for run in range(((file_n - 1) * n_runs), ((file_n - 1) * n_runs + n_runs)): 
\tprint(run)
\taa = timeit.default_timer()
\tresult_l[run - ((file_n - 1) * n_runs)] = netSTI.SIR_net_generator(run, 5000, years = 5, days = 14, 
\tgraph = graph, pEFoI = pEFoI, independent = indep, corr_scr = True, base_case = False, 
\ttrend = False, analysis_window = 2, output_netsum = True, output_net = False)
\tprint(timeit.default_timer() - aa)

with open('results/netout_' + graph + '_' + cor_or_not + '_' + str(file_n)+'.txt', 'w') as fout:
\tjson.dump(result_l, fout)
")}


pbs = function(graph, cor_or_not) {
  paste0("#!/bin/bash 
#PBS -l nodes=1:ppn=1,pmem=2500mb,walltime=5:00:00
#PBS -m abe
#PBS -M kaoxx085@umn.edu

cd /home/ennse/kaoxx085/networkSTI/
module load python
source networkSTI/bin/activate
python /home/ennse/kaoxx085/networkSTI/networkSTI/pyfile/netout_",graph, cor_or_not, "_$PBS_ARRAYID.py")
}


indep = "False"
if (indep == "False") {
  cor_or_not = "Corr"
} else {
  cor_or_not = "Uncorr"
}

graph = c("random", "community", "power_law", "empirical")

for(g in graph) {
  for(i in 1:10) {
    fn = paste0("/Users/szu-yukao/Documents/Network_structure_and_STI/networkSTI/msi files/netout_", g, cor_or_not, "_", i, ".py")
    writeLines(txt(i, g, indep = indep), fn)
  }
}


graph = c("random", "community", "power_law", "empirical")

for(g in graph) {
  fn = paste0("/Users/szu-yukao/Documents/Network_structure_and_STI/networkSTI/msi files/netout", g, cor_or_not, ".pbs")
  writeLines(pbs(g, cor_or_not), fn)
}





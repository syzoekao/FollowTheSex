###################################################
# summary of 5000 simulations 
##################################################

rm(list=ls())
library(data.table)
library(skimr)

mydata = lapply(c("randomDF", "communityDF", "powerLawDF"), function(g){
  mydata = data.table(read.csv(paste0("results/high notification/", g,".csv")), 
                      key = c("strategy"))
  mydata$tot_person_time = mydata$avg_durI*mydata$newI
  mydata$yUtil = mydata$Util/12
  null = aggregate(cbind(prevalence, newI, tot_person_time, ever_infected_node, avg_num_inf10, TotalCost, yUtil) ~ strategy, mydata[.("null")], mean)
  null$strategy = "NULL"
  null$pr = NA
  null.se = aggregate(cbind(prevalence, newI, tot_person_time, ever_infected_node, avg_num_inf10, TotalCost, yUtil) ~ strategy, mydata[.("null")], 
                      function(x) sd(x, na.rm = T)/(5000^0.5))
  null.se = null.se[, -1]
  colnames(null.se) = paste0("se.", colnames(null.se))
  null = cbind(null, null.se)
  
  PN = aggregate(cbind(prevalence, newI, tot_person_time, ever_infected_node, avg_num_inf10, TotalCost, yUtil) ~ strategy + p_treat_PN, mydata, mean)
  colnames(PN)[colnames(PN) %in% "p_treat_PN"] = "pr"
  PN.se = aggregate(cbind(prevalence, newI, tot_person_time, ever_infected_node, avg_num_inf10, TotalCost, yUtil) ~ strategy + p_treat_PN, mydata,  
                    function(x) sd(x, na.rm = T)/(5000^0.5))
  PN.se = PN.se[, -c(1,2)]
  colnames(PN.se) = paste0("se.", colnames(PN.se))
  PN = cbind(PN, PN.se)
  
  EPT = aggregate(cbind(prevalence, newI, tot_person_time, ever_infected_node, avg_num_inf10, TotalCost, yUtil) ~ strategy + p_treat_ept, mydata, mean)
  colnames(EPT)[colnames(EPT) %in% "p_treat_ept"] = "pr"
  EPT.se = aggregate(cbind(prevalence, newI, tot_person_time, ever_infected_node, avg_num_inf10, TotalCost, yUtil) ~ strategy + p_treat_ept, mydata,  
                     function(x) sd(x, na.rm = T)/(5000^0.5))
  EPT.se = EPT.se[, -c(1, 2)]
  colnames(EPT.se) = paste0("se.", colnames(EPT.se))
  EPT = cbind(EPT, EPT.se) 
  
  Tracing = aggregate(cbind(prevalence, newI, tot_person_time, ever_infected_node, avg_num_inf10, TotalCost, yUtil) ~ strategy + p_treat_tr, mydata, mean)
  colnames(Tracing)[colnames(Tracing) %in% "p_treat_tr"] = "pr"
  Tracing$strategy = "Tracing"
  Tracing.se = aggregate(cbind(prevalence, newI, tot_person_time, ever_infected_node, avg_num_inf10, TotalCost, yUtil) ~ strategy + p_treat_tr, mydata,  
                         function(x) sd(x, na.rm = T)/(5000^0.5))
  Tracing.se = Tracing.se[, -c(1, 2)]
  colnames(Tracing.se) = paste0("se.", colnames(Tracing.se))
  Tracing = cbind(Tracing, Tracing.se) 
  
  out = rbind(null, PN, EPT, Tracing)
  out$graph = g
  
  return(out)
})

names(mydata) = c("randomDF", "communityDF", "powerLawDF")

summary.outcomes = function(data, p_treat_PN = 0.71, p_treat_EPT = 0.79) {
  null = subset(data, strategy == "NULL")
  PN = subset(data, strategy == "PN" & pr == p_treat_PN)
  EPT = subset(data, strategy == "EPT" & pr == p_treat_EPT)
  Tracing = subset(data, strategy == "Tracing" & pr == max(c(p_treat_PN, p_treat_EPT)))
  out = rbind(null, PN, EPT, Tracing)
  return(out)
}

sum_out = lapply(c("randomDF", "communityDF", "powerLawDF"), 
                 function(g) summary.outcomes(mydata[[g]], p_treat_PN = 0.71, p_treat_EPT = 0.79))

print(sum_out)


###################################################
# 5000 simulations
##################################################

rm(list=ls())
library(data.table)


setEPS()
postscript('/Users/szu-yukao/Documents/Network_structure_and_STI/writing/CE comparison PN EPT Tracing (high notification 10 years).eps', height = 16, width = 20)
par(mar=c(6, 8, 5, 3), oma=c(1,1,3,0))
layout(matrix(c(1,2,3,4),byrow = T, ncol=2),heights=c(5, 5))

for (g in c("randomDF", "communityDF", "powerLawDF") ){
  mydata = data.table(read.csv(paste0("results/high notification/", g,".csv")))
  mydata[, 'Util'] = mydata[, 'Util']/12
  strategy = c("PN", "EPT", "contact tracing_degree")
  
  PN = aggregate(cbind(TotalCost, Util) ~ strategy + p_treat_PN, mydata, mean)
  EPT = aggregate(cbind(TotalCost, Util) ~ strategy + p_treat_ept, mydata, mean)
  Tracing = aggregate(cbind(TotalCost, Util) ~ strategy + p_treat_tr, mydata, mean)
  
  y.lab = round(seq(range(c(PN$TotalCost, EPT$TotalCost, Tracing$TotalCost))[1], 
                    range(c(PN$TotalCost, EPT$TotalCost, Tracing$TotalCost))[2], 
                    (range(c(PN$TotalCost, EPT$TotalCost, Tracing$TotalCost))[2]-
                       range(c(PN$TotalCost, EPT$TotalCost, Tracing$TotalCost))[1])/4), 3)
  x.lab = round(seq(range(c(PN$Util, EPT$Util, Tracing$Util))[1], 
                    range(c(PN$Util, EPT$Util, Tracing$Util))[2], 
                    (range(c(PN$Util, EPT$Util, Tracing$Util))[2]-
                       range(c(PN$Util, EPT$Util, Tracing$Util))[1])/4), 3)
  plot(PN$Util, PN$TotalCost, 
       type = "l", lty = 1, lwd = 3, 
       xlim = range(c(PN$Util, EPT$Util, Tracing$Util)), 
       ylim = range(c(PN$TotalCost, EPT$TotalCost, Tracing$TotalCost)), 
       xlab = NA, ylab = NA, cex.lab = 3, axes = FALSE)
  points(PN$Util, PN$TotalCost, 
         pch = 21, lwd = 3, bg = "white", cex = 3)
  lines(EPT$Util, EPT$TotalCost, lty = 2, lwd = 3)
  points(EPT$Util, EPT$TotalCost, 
         pch = 21, lwd = 3, bg = "white", cex = 3)
  lines(Tracing$Util, Tracing$TotalCost, lty = 3, lwd = 3)
  points(Tracing$Util, Tracing$TotalCost, 
         pch = 21, lwd = 3, bg = "white", cex = 3)
  
  axis(side = 1, at = x.lab, labels = F)
  text(x = x.lab, par("usr")[3]-0.0001, labels = round(x.lab, 3), pos = 1, xpd = TRUE, cex = 2)
  axis(side = 2, at = y.lab, labels = F) #labels = round(y.lab, 2), cex.axis = 2)
  text(y = y.lab+5, par("usr")[1]-0, labels = round(y.lab), srt = 0, pos = 2, xpd = TRUE, cex = 2)
  mtext("Costs",side = 2, line = 5.5, cex = 2.5)
  mtext("QALYs",side = 1, line = 4, cex = 2.5)
  mtext(g, side = 3, line = 3, cex = 3)
}

plot(c(0, 1.5), c(1, 1), 
     type = "l", lty = 1, lwd = 3, 
     xlim = c(0, 10), 
     ylim = c(0, 5), 
     xlab = NA, ylab = NA, cex.lab = 3, axes = FALSE)
points(0.75, 1, pch = 21, lwd = 3, bg = "white", cex = 3)
lines(c(0, 1.5), c(0.5, 0.5), lty = 2, lwd = 3)
points(0.75, 0.5, pch = 21, lwd = 3, bg = "white", cex = 3)
lines(c(0, 1.5), c(0, 0), lty = 3, lwd = 3)
points(0.75, 0, pch = 21, lwd = 3, bg = "white", cex = 3)
text(2, 1, "PN", cex = 2)
text(2, 0.5, "EPT", cex = 2)
text(2, 0, "Tracing", cex = 2)

dev.off()

# n_sel = sample(c(0:4999), 4000, replace = F)

mydata = lapply(c("randomDF", "communityDF", "powerLawDF"), function(g){
  mydata = data.table(read.csv(paste0("results/high notification/", g,".csv")), 
                      key = c("strategy"))
  # mydata = subset(mydata, run %in% n_sel)
  mydata[, 'Util'] = mydata[, 'Util']/12
  null = aggregate(cbind(TotalCost, Util) ~ strategy, mydata[.("null")], mean)
  null$strategy = "NULL"
  null$pr = NA
  PN = aggregate(cbind(TotalCost, Util) ~ strategy + p_treat_PN, mydata, mean)
  colnames(PN)[colnames(PN) %in% "p_treat_PN"] = "pr"
  EPT = aggregate(cbind(TotalCost, Util) ~ strategy + p_treat_ept, mydata, mean)
  colnames(EPT)[colnames(EPT) %in% "p_treat_ept"] = "pr"
  Tracing = aggregate(cbind(TotalCost, Util) ~ strategy + p_treat_tr, mydata, mean)
  colnames(Tracing)[colnames(Tracing) %in% "p_treat_tr"] = "pr"
  Tracing$strategy = "Tracing"
  out = rbind(null, PN, EPT, Tracing)
  out$graph = g
  return(out)
})

names(mydata) = c("randomDF", "communityDF", "powerLawDF")


CE.function = function(data, p_treat_PN, p_treat_EPT, wtp = 100000) {
  
  null.set = data[data$strategy == "NULL", ]
  tmp.PN = data[(data$strategy == "PN") & (data$pr == p_treat_PN), ]
  tmp.Tr = data[(data$strategy == "Tracing") & (data$pr == max(c(p_treat_PN, p_treat_EPT))), ]
  tmp.EPT = data[(data$strategy == "EPT") & (data$pr == p_treat_EPT), ]
  tmp.dat = rbind(null.set, tmp.EPT, tmp.PN, tmp.Tr)
  tmp.dat = data.frame(tmp.dat)
  tmp.dat = tmp.dat[order(tmp.dat$TotalCost), ]
  
  C.matrix = matrix(rep(tmp.dat$TotalCost, each = 4), nrow = 4, byrow = T) - matrix(rep(tmp.dat$TotalCost, 4), nrow = 4, byrow = T)
  U.matrix = matrix(rep(tmp.dat$Util, each = 4), nrow = 4, byrow = T) - matrix(rep(tmp.dat$Util, 4), nrow = 4, byrow = T)
  colnames(C.matrix) = rownames(C.matrix) = colnames(U.matrix) = rownames(U.matrix) = tmp.dat$strategy
  CE.matrix = C.matrix/U.matrix
  CE.matrix = lower.tri(CE.matrix, diag = FALSE)*CE.matrix
  CE.matrix[is.nan(CE.matrix)] = 0
  
  # check strongly dominated strategy and elimiate the strategy
  eliminate_set = rowSums(CE.matrix<0)
  eliminate_strategy = names(eliminate_set)[eliminate_set > 0]
  
  # red.CE.matrix = CE.matrix[!(rownames(CE.matrix) %in% eliminate_strategy), ]
  CE.matrix[CE.matrix<=0] = NA
  
  col.min = apply(CE.matrix, 2, min, na.rm = T)
  col.min[is.infinite(col.min)] = NA
  st.selected = do.call(rbind, lapply(c(1:(ncol(CE.matrix)-1)), function(x) {
    st = rownames(CE.matrix)[which(CE.matrix[, x] == col.min[x])]
    if(length(st)==0) {
      st = "NA"
      ce.val = "NA"
    } else {
      ce.val = CE.matrix[st,x]
    }
    out = c(st, ce.val)
    names(out) = c("st", "ce.val")
    return(out)
  }))
  
  st.selected = data.frame(st.selected)
  st.selected = st.selected[!duplicated(st.selected[, "st"]), ]
  
  ce.df = tmp.dat
  ce.df$ICER = st.selected$ce.val[match(ce.df$st, st.selected$st)]
  ce.df$ICER = as.character(ce.df$ICER)
  ce.df$ICER[1] = "NA"
  ce.df$ICER[match(eliminate_strategy, ce.df$st)] = "strongly dominated"
  ce.df$ICER[is.na(ce.df$ICER)] = "weakly dominated"
  
  val_vec = as.numeric(as.character(ce.df$ICER))
  opt.st = ce.df$strategy[ce.df$ICER==as.character(max(val_vec[val_vec<wtp], na.rm= T))]
  if(length(opt.st) == 0) { opt.st = 'NULL'}
  
  return(list(ce.df = ce.df, opt = opt.st))
}


random_mat = matrix(NA, nrow = 10, ncol = 10)
colnames(random_mat) = seq(0.1, 1, 0.1)
rownames(random_mat) = rev(seq(0.1, 1, 0.1))
for(j in c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1)) {
  for(i in c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1)) {
    out = CE.function(data = mydata[['randomDF']], p_treat_PN = j, p_treat_EPT = i)
    random_mat[paste0(j), paste0(i)] = out[["opt"]]
  }
}

random_mat = ifelse(random_mat == "NULL", 0, 
                    ifelse(random_mat == "PN", 1, 
                           ifelse(random_mat == "EPT", 2, 3)))


community_mat = matrix(NA, nrow = 10, ncol = 10)
colnames(community_mat) = seq(0.1, 1, 0.1)
rownames(community_mat) = rev(seq(0.1, 1, 0.1))
for(j in c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1)) {
  for(i in c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1)) {
    out = CE.function(data = mydata[['communityDF']], p_treat_PN = j, p_treat_EPT = i)
    community_mat[paste0(j), paste0(i)] = out[["opt"]]
  }
}

community_mat = ifelse(community_mat == "NULL", 0, 
                       ifelse(community_mat == "PN", 1, 
                              ifelse(community_mat == "EPT", 2, 3)))


powerLaw_mat = matrix(NA, nrow = 10, ncol = 10)
colnames(powerLaw_mat) = seq(0.1, 1, 0.1)
rownames(powerLaw_mat) = rev(seq(0.1, 1, 0.1))
for(j in c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1)) {
  for(i in c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1)) {
    out = CE.function(data = mydata[['powerLawDF']], p_treat_PN = j, p_treat_EPT = i)
    powerLaw_mat[paste0(j), paste0(i)] = out[["opt"]]
  }
}

powerLaw_mat = ifelse(powerLaw_mat == "NULL", 0, 
                      ifelse(powerLaw_mat == "PN", 1, 
                             ifelse(powerLaw_mat == "EPT", 2, 3)))

library(raster)

png('/Users/szu-yukao/Documents/Network_structure_and_STI/writing/sensitivity plot_5000 simulation (high notification 10 years).png', height = 1500, width = 2000)

par(mar=c(6, 8, 5, 3), oma=c(1,1,3,0))
layout(matrix(c(1,2,3,4),byrow = T, ncol=2),heights=c(5, 5))
x.lab = seq(0.1, 1, 0.1)
y.lab = seq(0.1, 1, 0.1)

breakpoints <- c(0, 0.5, 1.5, 2.5)
colors <- c("gray80", "royalblue1", "limegreen", "deeppink")
plot(raster(random_mat),breaks=breakpoints,col=colors,legend=FALSE, axes=F)
axis(side = 1, at = x.lab-0.05, labels = F)
text(x = x.lab-0.05, par("usr")[3]-0.01, labels = x.lab, pos = 1, xpd = TRUE, cex = 2)
axis(side = 2, at = y.lab-0.05, labels = F)
text(y = y.lab-0.05, par("usr")[1]-0.01, labels = y.lab, srt = 0, pos = 2, xpd = TRUE, cex = 2)
points(0.8, 0.67, pch = "*", col = "deeppink", cex = 8, lwd = 3)
mtext(expression(phi^{EPT}),side = 1, line = 4, cex = 2)
mtext(expression(phi^{PN}),side = 2, line = 3, cex = 2)
mtext("(A) Random network",side = 3, line = 2, cex = 2.5)

plot(raster(community_mat),breaks=breakpoints,col=colors,legend=FALSE, axes=F)
axis(side = 1, at = x.lab-0.05, labels = F)
text(x = x.lab-0.05, par("usr")[3]-0.01, labels = x.lab, pos = 1, xpd = TRUE, cex = 2)
axis(side = 2, at = y.lab-0.05, labels = F)
text(y = y.lab-0.05, par("usr")[1]-0.01, labels = y.lab, srt = 0, pos = 2, xpd = TRUE, cex = 2)
points(0.8, 0.67, pch = "*", col = "deeppink", cex = 8, lwd = 3)
mtext(expression(phi^{EPT}),side = 1, line = 4, cex = 2)
mtext(expression(phi^{PN}),side = 2, line = 3, cex = 2)
mtext("(B) Community network",side = 3, line = 2, cex = 2.5)

plot(raster(powerLaw_mat),breaks=breakpoints,col=colors,legend=FALSE, axes=F)
axis(side = 1, at = x.lab-0.05, labels = F)
text(x = x.lab-0.05, par("usr")[3]-0.01, labels = x.lab, pos = 1, xpd = TRUE, cex = 2)
axis(side = 2, at = y.lab-0.05, labels = F)
text(y = y.lab-0.05, par("usr")[1]-0.01, labels = y.lab, srt = 0, pos = 2, xpd = TRUE, cex = 2)
points(0.8, 0.67, pch = "*", col = "deeppink", cex = 8, lwd = 3)
mtext(expression(phi^{EPT}),side = 1, line = 4, cex = 2)
mtext(expression(phi^{PN}),side = 2, line = 3, cex = 2)
mtext("(C) Scale-free network",side = 3, line = 2, cex = 2.5)

plot(3, 1, pch = 22,
     xlim = c(0, 10), 
     ylim = c(0, 10), 
     xlab = NA, ylab = NA, cex = 5, axes = FALSE, 
     col = "gray80", bg = "gray80")
text(3.5, 1, "Routine screening alone", cex = 3, adj = c(0,0.5))
points(3, 2, pch = 22, cex = 5, col = "royalblue1", bg = "royalblue1")
text(3.5, 2, "PN", cex = 3, adj = c(0,0.5))
points(3, 3, pch = 22, cex = 5, col = "limegreen", bg = "limegreen")
text(3.5, 3, "EPT", cex = 3, adj = c(0,0.5))
points(3, 4, pch = 22, cex = 5, col = "pink", bg = "pink")
text(3.5, 4, "Contact tracing", cex = 3, adj = c(0,0.5))
points(3, 5, pch = "*", cex = 8, col = "deeppink", bg = "deeppink")
text(3.5, 5, "high notification", cex = 3, adj = c(0,0.5))

dev.off()



##############################################
# Estimation
##############################################

rm(list=ls())
library(parallel)
library(R2jags)
library(coda)
library(mcmc)
library(data.table)

readData = function(g, strategy) {
  mydata = data.table(read.csv(paste0("results/high notification/", g,"DF.csv")), 
                      key = c("strategy"))
  if(strategy == "EPT") {
    colsel = c("strategy", "Util", "TotalCost", "p_treat_ept")
  } else if (strategy == "PN") {
    colsel = c("strategy", "Util", "TotalCost", "p_treat_PN")
  } else {
    colsel = c("strategy", "Util", "TotalCost", "p_treat_tr")
  }
  out = mydata[strategy, ..colsel]
  out[,"Util"] = out[,"Util"]/12
  colnames(out) = c("strategy", "Util", "TotalCost", "pr")
  
  agg = aggregate(cbind(TotalCost, Util)~pr, out, mean)
  i.C = min(agg$TotalCost)
  s.C = (max(agg$TotalCost)-min(agg$TotalCost))/(max(agg$pr)-min(agg$pr))
  i.U = min(agg$Util)
  s.U = (max(agg$Util)-min(agg$Util))/(max(agg$pr)-min(agg$pr))
  
  agg = aggregate(cbind(TotalCost, Util)~pr, out, sd)
  sd.i.C = agg$TotalCost[agg$pr == 0.1]
  sd.s.C = mean(agg$TotalCost)
  sd.i.U = agg$Util[agg$pr == 0.1]
  sd.s.U = mean(agg$Util)
  
  mean_stat = c(i.C, s.C, i.U, s.U)
  names(mean_stat) = c("int.C", "slo.C", "int.U", "slo.U")
  sd_stat = c(sd.i.C, sd.s.C, sd.i.U, sd.s.U)
  names(sd_stat) = c("sd.int.C", "sd.slo.C", "sd.int.U", "sd.slo.U")
  
  return(list(data = out, mean_stat = mean_stat, sd_stat = sd_stat))
}

model <- function() {
  
  for(i in 1:n) {
    Cost[i] ~ dnorm(mu[i, 1], tau[1])
    mu[i, 1] <- beta0[1] + beta[1]*pr[i]
    
    Util[i] ~ dnorm(mu[i, 2], tau[2])
    mu[i, 2] <- beta0[2] + beta[2]*pr[i]
  }
  
  # for(i in 1:Nnew) {
  #   CostHat[i] ~ dnorm(mu.c[i], tau[1])
  #   mu.c[i] <- beta0[1] + beta[1]*pr.new[i]
  #   UtilHat[i] ~ dnorm(mu.u[i], tau[2])
  #   mu.u[i] <- beta0[2] + beta[2]*pr.new[i]
  # }
  
  tau.int.C = 1/(pow(sd.int.C,2))
  tau.slo.C = 1/(pow(sd.slo.C,2))
  tau.int.U = 1/(pow(sd.int.U,2))
  tau.slo.U = 1/(pow(sd.slo.U,2))
  
  beta0[1] ~ dnorm(int.C, tau.int.C)
  beta[1] ~ dnorm(slo.C, tau.slo.C)
  beta0[2] ~ dnorm(int.U, tau.int.U)
  beta[2] ~ dnorm(slo.U, tau.slo.U)
  
  tau[1] ~ dgamma(0.001, 0.001)
  sigma[1] <- 1/sqrt(tau[1])
  tau[2] ~ dgamma(0.001, 0.001)
  sigma[2] <- 1/sqrt(tau[2])
  
}

for(g in c("random", "community", "powerLaw")) {
  print(g)
  for(st in c("PN", "EPT", "contact tracing_degree")){
    print(st)
    
    mydata = readData(g, st)
    
    pr = seq(0.01, 1, 0.01)
    missingValues = data.table(strategy = rep(st, length(pr)), 
                               Util = rep(NA, length(pr)), 
                               TotalCost = rep(NA, length(pr)), 
                               pr = pr, 
                               key = "pr")
    
    data = list(n = nrow(mydata$data), 
                pr = mydata$data[, pr], 
                Cost = mydata$data[, TotalCost], 
                Util = mydata$data[, Util], 
                Nnew = nrow(missingValues), 
                # pr.new = missingValues[,pr], 
                int.C = mydata$mean_stat[['int.C']], 
                slo.C = mydata$mean_stat[['slo.C']], 
                int.U = mydata$mean_stat[['int.U']], 
                slo.U = mydata$mean_stat[['slo.U']], 
                sd.int.C = mydata$sd_stat[['sd.int.C']], 
                sd.slo.C = mydata$sd_stat[['sd.slo.C']], 
                sd.int.U = mydata$sd_stat[['sd.int.U']], 
                sd.slo.U = mydata$sd_stat[['sd.slo.U']])
    
    aa = Sys.time()
    output <- jags(data = data, model.file = model, 
                   parameters=c("beta0", "beta", "sigma"), # "CostHat", 'UtilHat'), 
                   n.chains = 1, n.iter=35000,n.burnin=5000,n.thin=1)
    print(Sys.time()-aa)
    
    # plot(apply(output$BUGSoutput$sims.array[,,paste0("CostHat[", c(1:100), "]")], 2, mean), type = "l")
    # plot(apply(output$BUGSoutput$sims.array[,,paste0("UtilHat[", c(1:100), "]")], 2, mean), type = "l")
    # plot(apply(output$BUGSoutput$sims.array[,,paste0("UtilHat[", c(1:100), "]")], 2, mean), 
    #      apply(output$BUGSoutput$sims.array[,,paste0("CostHat[", c(1:100), "]")], 2, mean), 
    #      type = "l")
    
    # U.lm = lm(Util ~ pr, data = mydata$data)
    # C.lm = lm(TotalCost ~ pr, data = mydata$data)
    # new.pr = data.frame(pr = seq(0.01, 1, 0.01))
    # pred.U = replicate(5000, predict(U.lm, newdata = new.pr)+rnorm(nrow(new.pr), 0, summary(U.lm)$sigma))
    # pred.C = replicate(5000, predict(C.lm, newdata = new.pr)+rnorm(nrow(new.pr), 0, summary(C.lm)$sigma))
    
    # mean.U = apply(output$BUGSoutput$sims.array[,,paste0("UtilHat[", c(1:100), "]")], 2, mean)
    # mean.C = apply(output$BUGSoutput$sims.array[,,paste0("CostHat[", c(1:100), "]")], 2, mean)
    
    if(st == "contact tracing_degree") {st = "Tracing"}
    # ce.data = data.frame(graph = g, strategy = st, pr = pr, 
    #                      TotalCost = mean.C, Util = mean.U)
    write.csv(output$BUGSoutput$summary, paste0("results/high notification/bayes_estimate_", g, "_", st, ".csv"))
    # write.csv(ce.data, paste0("results/high notification/bayes_sim_", g, "_", st, ".csv"), row.names = F)
  }
}


###########################################
# smooth function
###########################################


CE.function = function(data, p_treat_PN, p_treat_EPT, wtp = 100000) {
  # print(p_treat_PN)
  # print(p_treat_EPT)
  # print(max(c(p_treat_PN, p_treat_EPT)))
  
  null.set = data[data$strategy == "NULL", ]
  tmp.PN = data[(data$strategy == "PN") & (data$pr == round(p_treat_PN,2)), ]
  # tmp.Tr = data[(data$strategy == "Tracing") & (data$pr == round(max(c(p_treat_PN, p_treat_EPT)),2)), ]
  tmp.Tr = data[(data$strategy == "Tracing") & (data$pr == round(max(c(p_treat_PN, p_treat_EPT)),2)), ]
  tmp.EPT = data[(data$strategy == "EPT") & (data$pr == round(p_treat_EPT,2)), ]
  tmp.dat = rbind(null.set, tmp.EPT, tmp.PN, tmp.Tr)
  tmp.dat = data.frame(tmp.dat)
  tmp.dat = tmp.dat[order(tmp.dat$TotalCost), ]
  
  C.matrix = matrix(rep(tmp.dat$TotalCost, each = 4), nrow = 4, byrow = T) - matrix(rep(tmp.dat$TotalCost, 4), nrow = 4, byrow = T)
  U.matrix = matrix(rep(tmp.dat$Util, each = 4), nrow = 4, byrow = T) - matrix(rep(tmp.dat$Util, 4), nrow = 4, byrow = T)
  colnames(C.matrix) = rownames(C.matrix) = colnames(U.matrix) = rownames(U.matrix) = tmp.dat$strategy
  CE.matrix = C.matrix/U.matrix
  CE.matrix = lower.tri(CE.matrix, diag = FALSE)*CE.matrix
  CE.matrix[is.nan(CE.matrix)] = 0
  
  # check strongly dominated strategy and elimiate the strategy
  eliminate_set = rowSums(CE.matrix<0)
  eliminate_strategy = names(eliminate_set)[eliminate_set > 0]
  
  # red.CE.matrix = CE.matrix[!(rownames(CE.matrix) %in% eliminate_strategy), ]
  CE.matrix[CE.matrix<=0] = NA
  
  col.min = apply(CE.matrix, 2, min, na.rm = T)
  col.min[is.infinite(col.min)] = NA
  st.selected = do.call(rbind, lapply(c(1:(ncol(CE.matrix)-1)), function(x) {
    st = rownames(CE.matrix)[which(CE.matrix[, x] == col.min[x])]
    if(length(st)==0) {
      st = "NA"
      ce.val = "NA"
    } else {
      ce.val = CE.matrix[st,x]
    }
    out = c(st, ce.val)
    names(out) = c("st", "ce.val")
    return(out)
  }))
  
  st.selected = data.frame(st.selected)
  st.selected = st.selected[!duplicated(st.selected[, "st"]), ]
  
  ce.df = tmp.dat
  ce.df$ICER = st.selected$ce.val[match(ce.df$st, st.selected$st)]
  ce.df$ICER = as.character(ce.df$ICER)
  ce.df$ICER[1] = "NA"
  ce.df$ICER[match(eliminate_strategy, ce.df$st)] = "strongly dominated"
  ce.df$ICER[is.na(ce.df$ICER)] = "weakly dominated"
  
  val_vec = as.numeric(as.character(ce.df$ICER))
  opt.st = ce.df$strategy[ce.df$ICER==as.character(max(val_vec[val_vec<wtp], na.rm= T))]
  
  if(length(opt.st) == 0) {
    opt.st = "NULL"
  } else {
    opt.st = opt.st
  }
  
  return(list(ce.df = ce.df, opt = opt.st))
}


mydata = lapply(c("random", "community", "powerLaw"), function(g){
  nullData = read.csv(paste0("results/base case/", g,"DF.csv"))
  nullData = subset(nullData, strategy == "null", select = c("strategy", "TotalCost", "Util"))
  nullData = aggregate(cbind(TotalCost, Util)~strategy, nullData, mean)
  nullData$strategy = "NULL"
  nullData$graph = g
  nullData$pr = NA
  nullData$Util = nullData$Util/12
  pr = round(seq(0, 1, 0.01), 2)
  
  out = nullData
  for (st in c("PN", "EPT", "Tracing")){
    tempDF = read.csv(paste0("results/base case/bayes_estimate_",g,"_", st, ".csv"))
    betas = tempDF[tempDF$X %in% c("beta[1]", "beta[2]", "beta0[1]", "beta0[2]"), "mean"]
    TotalCost = betas[3]+betas[1]*pr
    Util = betas[4]+betas[2]*pr
    
    stDF = data.frame(strategy = st, TotalCost = TotalCost, Util = Util, graph = g, 
                      pr = pr)
    out = rbind(out, stDF)
  }
  
  out$graph = g
  return(out)
})

names(mydata) = c("randomDF", "communityDF", "powerLawDF")


wtp = 100000

random_mat = matrix(NA, nrow = 100, ncol = 100)
colnames(random_mat) = seq(0.01, 1, 0.01)
rownames(random_mat) = rev(seq(0.01, 1, 0.01))
for(j in seq(0.01, 1, 0.01)) {
  for(i in seq(0.01, 1, 0.01)) {
    out = CE.function(data = mydata[['randomDF']], p_treat_PN = j, p_treat_EPT = i, wtp = wtp)
    random_mat[paste0(j), paste0(i)] = out[["opt"]]
  }
}

random_mat = ifelse(random_mat == "NULL", 0, 
                    ifelse(random_mat == "PN", 1, 
                           ifelse(random_mat == "EPT", 2, 3)))


community_mat = matrix(NA, nrow = 100, ncol = 100)
colnames(community_mat) = seq(0.01, 1, 0.01)
rownames(community_mat) = rev(seq(0.01, 1, 0.01))
for(j in seq(0.01, 1, 0.01)) {
  for(i in seq(0.01, 1, 0.01)) {
    out = CE.function(data = mydata[['communityDF']], p_treat_PN = j, p_treat_EPT = i, wtp = wtp)
    community_mat[paste0(j), paste0(i)] = out[["opt"]]
  }
}

community_mat = ifelse(community_mat == "NULL", 0, 
                       ifelse(community_mat == "PN", 1, 
                              ifelse(community_mat == "EPT", 2, 3)))


powerLaw_mat = matrix(NA, nrow = 100, ncol = 100)
colnames(powerLaw_mat) = seq(0.01, 1, 0.01)
rownames(powerLaw_mat) = rev(seq(0.01, 1, 0.01))
for(j in seq(0.01, 1, 0.01)) {
  for(i in seq(0.01, 1, 0.01)) {
    out = CE.function(data = mydata[['powerLawDF']], p_treat_PN = j, p_treat_EPT = i, wtp = wtp)
    powerLaw_mat[paste0(j), paste0(i)] = out[["opt"]]
  }
}

powerLaw_mat = ifelse(powerLaw_mat == "NULL", 0, 
                      ifelse(powerLaw_mat == "PN", 1, 
                             ifelse(powerLaw_mat == "EPT", 2, 3)))


cropWithRowCol <- function(r, rows, cols) {
  cc <- cellFromRowColCombine(r, rownr=rows, colnr=cols)
  crop(r, rasterFromCells(r, cc, values=FALSE))
}

library(raster)
setEPS()
# postscript('/Users/szu-yukao/Documents/Network_structure_and_STI/writing/sensitivity plot_smooth_10000 (base case 10 years).eps', height = 15, width = 20)
png('/Users/szu-yukao/Documents/Network_structure_and_STI/networkSTI/results/RRsummary/sensitivity plot_smooth_100000 (base case 10 years).png', height = 1500, width = 2000)

par(mar=c(6, 8, 5, 3), oma = c(0, 0, 0, 0)) # oma=c(1,1,3,0))
layout(matrix(c(1,2,3,4),byrow = T, ncol=2),heights=c(5, 5))
x.lab = seq(0, 1, 0.1)
y.lab = seq(0, 1, 0.1)

breakpoints <- c(0, 0.5, 1.5, 2.5)
colors <- c("gray80", "royalblue1", "limegreen", "deeppink")
ras = raster(random_mat)
r <- cropWithRowCol(ras, nrow(ras) - 100:0, 1:100)
w <- ncol(r)/max(dim(r))
h <- nrow(r)/max(dim(r))
# dev.new(width = 5*w, height = 5*h)
plot(ras,breaks=breakpoints,col=colors,legend=FALSE, axes=F)
axis(side = 1, at = x.lab, labels = F)
text(x = x.lab, par("usr")[3]-0.01, labels = x.lab, pos = 1, xpd = TRUE, cex = 2)
axis(side = 2, at = y.lab, labels = F)
text(y = y.lab, par("usr")[1]-0.01, labels = y.lab, srt = 0, pos = 2, xpd = TRUE, cex = 2)
# abline(coef = c(0, 1), col = 'black', lty = 3, lwd = 1.5)
points(0.79, 0.71, pch = "*", col = "deeppink", cex = 8, lwd = 3)
mtext('EPT partner compliance',side = 1, line = 4, cex = 2.5)
mtext('PN partner compliance',side = 2, line = 4.5, cex = 2.5)
mtext("(A) Random network",side = 3, line = 2, cex = 3)

ras = raster(community_mat)
# r <- cropWithRowCol(ras, nrow(ras) - 100:0, 1:100)
# w <- ncol(r)/max(dim(r))
# h <- nrow(r)/max(dim(r))
# dev.new(width = 5*w, height = 5*h)
plot(ras,breaks=breakpoints,col=colors,legend=FALSE, axes=F, asp=1)
axis(side = 1, at = x.lab, labels = F)
text(x = x.lab, par("usr")[3]-0.01, labels = x.lab, pos = 1, xpd = TRUE, cex = 2)
axis(side = 2, at = y.lab, labels = F)
text(y = y.lab, par("usr")[1]-0.01, labels = y.lab, srt = 0, pos = 2, xpd = TRUE, cex = 2)
# abline(coef = c(0, 1), col = 'black', lty = 3, lwd = 1.5)
points(0.79, 0.71, pch = "*", col = "deeppink", cex = 8, lwd = 3)
mtext('EPT partner compliance',side = 1, line = 4, cex = 2.5)
mtext('PN partner compliance',side = 2, line = 4.5, cex = 2.5)
mtext("(B) Community network",side = 3, line = 2, cex = 3)

ras = raster(powerLaw_mat)
# r <- cropWithRowCol(ras, nrow(ras) - 100:0, 1:100)
# w <- ncol(r)/max(dim(r))
# h <- nrow(r)/max(dim(r))
# dev.new(width = 5*w, height = 5*h)
plot(ras,breaks=breakpoints,col=colors,legend=FALSE, axes=F, asp=1)
axis(side = 1, at = x.lab, labels = F)
text(x = x.lab, par("usr")[3]-0.01, labels = x.lab, pos = 1, xpd = TRUE, cex = 2)
axis(side = 2, at = y.lab, labels = F)
text(y = y.lab, par("usr")[1]-0.01, labels = y.lab, srt = 0, pos = 2, xpd = TRUE, cex = 2)
# abline(coef = c(0, 1), col = 'black', lty = 3, lwd = 1.5)
points(0.79, 0.71, pch = "*", col = "deeppink", cex = 8, lwd = 3)
mtext('EPT partner compliance',side = 1, line = 4, cex = 2.5)
mtext('PN partner compliance',side = 2, line = 4.5, cex = 2.5)
mtext("(C) Scale-free network",side = 3, line = 2, cex = 3)

plot(3, 1, pch = 22,
     xlim = c(0, 10), 
     ylim = c(0, 10), 
     xlab = NA, ylab = NA, cex = 5, axes = FALSE, 
     col = "gray80", bg = "gray80")
text(3.5, 1, "Routine screening alone", cex = 3, adj = c(0,0.5))
points(3, 2, pch = 22, cex = 5, col = "royalblue1", bg = "royalblue1")
text(3.5, 2, "PN", cex = 3, adj = c(0,0.5))
points(3, 3, pch = 22, cex = 5, col = "limegreen", bg = "limegreen")
text(3.5, 3, "EPT", cex = 3, adj = c(0,0.5))
points(3, 4, pch = 22, cex = 5, col = "pink", bg = "pink")
text(3.5, 4, "Contact tracing", cex = 3, adj = c(0,0.5))
points(3, 5, pch = "*", cex = 8, col = "deeppink", bg = "deeppink")
text(3.5, 5, "Base case", cex = 3, adj = c(0,0.5))

dev.off()


# optimal is routine screening alone: 
thres.screening2PN = rep(NA, 3)
for (i in c(100:50)) {
  temp = random_mat[i, 1] - random_mat[i-1, 1]
  if (temp < 0) {
    thres = as.numeric(row.names(random_mat)[i-1])
    break
  }
}
thres.screening2PN[1]= thres

for (i in c(100:50)) {
  temp = community_mat[i, 1] - community_mat[i-1, 1]
  if (temp < 0) {
    thres = as.numeric(row.names(community_mat)[i-1])
    break
  }
}
thres.screening2PN[2]= thres

for (i in c(100:50)) {
  temp = powerLaw_mat[i, 1] - powerLaw_mat[i-1, 1]
  if (temp < 0) {
    thres = as.numeric(row.names(powerLaw_mat)[i-1])
    break
  }
}
thres.screening2PN[3]= thres

# optimal change from cheaper to more expensive: 
thres.random = data.frame(cheaper = rownames(random_mat), EPT = NA)
for (i in rownames(random_mat)) {
  for (j in c(1:99)) {
    temp = random_mat[i, j] - random_mat[i, j+1]
    # print(temp)
    if (temp < 0) {
      thres.random[thres.random$cheaper == i, "EPT"]=as.numeric(colnames(random_mat)[j])
    }
  }
}
thres.random$cheaper = as.numeric(as.character(thres.random$cheaper))
thres.random$diff = thres.random$EPT-thres.random$cheaper

thres.community = data.frame(cheaper = rownames(community_mat), EPT = NA)
for (i in rownames(community_mat)) {
  for (j in c(1:99)) {
    temp = community_mat[i, j] - community_mat[i, j+1]
    # print(temp)
    if (temp < 0) {
      thres.community[thres.community$cheaper == i, "EPT"]=as.numeric(colnames(community_mat)[j])
    }
  }
}
thres.community$cheaper = as.numeric(as.character(thres.community$cheaper))
thres.community$diff = thres.community$EPT-thres.community$cheaper

thres.powerLaw = data.frame(cheaper = rownames(powerLaw_mat), EPT = NA)
for (i in rownames(powerLaw_mat)) {
  for (j in c(1:99)) {
    temp = powerLaw_mat[i, j] - powerLaw_mat[i, j+1]
    # print(temp)
    if (temp < 0) {
      thres.powerLaw[thres.powerLaw$cheaper == i, "EPT"]=as.numeric(colnames(powerLaw_mat)[j])
    }
  }
}
thres.powerLaw$cheaper = as.numeric(as.character(thres.powerLaw$cheaper))
thres.powerLaw$diff = thres.powerLaw$EPT-thres.powerLaw$cheaper

thres.random = thres.random[order(thres.random$cheaper), ]
thres.community = thres.community[order(thres.community$cheaper), ]
thres.powerLaw = thres.powerLaw[order(thres.powerLaw$cheaper), ]

write.csv(thres.random, "/Users/szu-yukao/Documents/Network_structure_and_STI/writing/thres_random (high notification 10 years).csv", row.names = F)
write.csv(thres.community, "/Users/szu-yukao/Documents/Network_structure_and_STI/writing/thres_community (high notification 10 years).csv", row.names = F)
write.csv(thres.powerLaw, "/Users/szu-yukao/Documents/Network_structure_and_STI/writing/thres_powerLaw (high notification 10 years).csv", row.names = F)


out = data.frame(strategy = character(), TotalCost=numeric(), Util=numeric(), graph=character(), 
                 pr = numeric())
for(g in c("random", "community", "powerLaw")){
  for (st in c("PN", "EPT", "Tracing")){
    tempDF = read.csv(paste0("results/high notification/bayes_estimate_",g,"_", st, ".csv"))
    betas = tempDF[tempDF$X %in% c("beta[1]", "beta[2]", "beta0[1]", "beta0[2]"), "mean"]
    TotalCost = betas[3]+betas[1]*pr
    Util = betas[4]+betas[2]*pr
    
    stDF = data.frame(strategy = rep(st, length(pr)), TotalCost = TotalCost, 
                      Util = Util, graph = rep(g, length(pr)), pr = pr)
    out = rbind(out, stDF)
  }
}

setEPS()
postscript('/Users/szu-yukao/Documents/Network_structure_and_STI/writing/CE comparison PN EPT Tracing (high notification 10 years).eps', height = 16, width = 20)
par(mar=c(6, 8, 5, 3), oma=c(1,1,3,0))
layout(matrix(c(1,2,3,4),byrow = T, ncol=2),heights=c(5, 5))

for (g in c("random", "community", "powerLaw") ){
  mydata = data.table(read.csv(paste0("results/high notification/", g,"DF.csv")))
  mydata[, 'Util'] = mydata[, 'Util']/12
  strategy = c("PN", "EPT", "contact tracing_degree")
  
  PN = aggregate(cbind(TotalCost, Util) ~ strategy + p_treat_PN, mydata, mean)
  EPT = aggregate(cbind(TotalCost, Util) ~ strategy + p_treat_ept, mydata, mean)
  Tracing = aggregate(cbind(TotalCost, Util) ~ strategy + p_treat_tr, mydata, mean)
  
  # PN2 = read.csv(paste0("results/high notification/bayes_sim_",g,"_PN.csv"))
  # EPT2 = read.csv(paste0("results/high notification/bayes_sim_",g,"_EPT.csv"))
  # Tracing2 = read.csv(paste0("results/high notification/bayes_sim_",g,"_Tracing.csv"))
  
  PN3 = subset(out, strategy == "PN" & graph == g)
  EPT3 = subset(out, strategy == "EPT" & graph == g)
  Tracing3 = subset(out, strategy == "Tracing" & graph == g)
  
  y.lab = round(seq(range(c(PN$TotalCost, EPT$TotalCost, Tracing$TotalCost))[1], 
                    range(c(PN$TotalCost, EPT$TotalCost, Tracing$TotalCost))[2], 
                    (range(c(PN$TotalCost, EPT$TotalCost, Tracing$TotalCost))[2]-
                       range(c(PN$TotalCost, EPT$TotalCost, Tracing$TotalCost))[1])/4), 3)
  x.lab = round(seq(range(c(PN$Util, EPT$Util, Tracing$Util))[1], 
                    range(c(PN$Util, EPT$Util, Tracing$Util))[2], 
                    (range(c(PN$Util, EPT$Util, Tracing$Util))[2]-
                       range(c(PN$Util, EPT$Util, Tracing$Util))[1])/4), 3)
  plot(PN$Util, PN$TotalCost, 
       type = "l", lty = 1, lwd = 3, 
       xlim = range(c(PN$Util, EPT$Util, Tracing$Util)), 
       ylim = range(c(PN$TotalCost, EPT$TotalCost, Tracing$TotalCost)), 
       xlab = NA, ylab = NA, cex.lab = 3, axes = FALSE)
  # lines(PN2$Util, PN2$TotalCost, lty = 1, lwd = 1, col = "royalblue")
  lines(PN3$Util, PN3$TotalCost, lty = 1, lwd = 3, col = "limegreen")
  points(PN$Util, PN$TotalCost, 
         pch = 21, lwd = 3, bg = "white", cex = 3)
  # lines(EPT2$Util, EPT2$TotalCost, lty = 1, lwd = 1, col = "royalblue")
  lines(EPT3$Util, EPT3$TotalCost, lty = 1, lwd = 3, col = "limegreen")
  lines(EPT$Util, EPT$TotalCost, lty = 2, lwd = 3)
  points(EPT$Util, EPT$TotalCost, 
         pch = 21, lwd = 3, bg = "white", cex = 3)
  # lines(Tracing2$Util, Tracing2$TotalCost, lty = 1, lwd = 1, col = "royalblue")
  lines(Tracing3$Util, Tracing3$TotalCost, lty = 1, lwd = 3, col = "limegreen")
  lines(Tracing$Util, Tracing$TotalCost, lty = 3, lwd = 3)
  points(Tracing$Util, Tracing$TotalCost, 
         pch = 21, lwd = 3, bg = "white", cex = 3)
  
  axis(side = 1, at = x.lab, labels = F)
  text(x = x.lab, par("usr")[3]-0.0001, labels = round(x.lab, 3), pos = 1, xpd = TRUE, cex = 2)
  axis(side = 2, at = y.lab, labels = F) #labels = round(y.lab, 2), cex.axis = 2)
  text(y = y.lab+5, par("usr")[1]-0, labels = round(y.lab), srt = 0, pos = 2, xpd = TRUE, cex = 2)
  mtext("Costs",side = 2, line = 5.5, cex = 2.5)
  mtext("QALYs",side = 1, line = 4, cex = 2.5)
  mtext(g, side = 3, line = 3, cex = 3)
}

plot(c(0, 1.5), c(1, 1), 
     type = "l", lty = 1, lwd = 3, 
     xlim = c(0, 10), 
     ylim = c(0, 5), 
     xlab = NA, ylab = NA, cex.lab = 3, axes = FALSE)
points(0.75, 1, pch = 21, lwd = 3, bg = "white", cex = 3)
lines(c(0, 1.5), c(0.5, 0.5), lty = 2, lwd = 3)
points(0.75, 0.5, pch = 21, lwd = 3, bg = "white", cex = 3)
lines(c(0, 1.5), c(0, 0), lty = 3, lwd = 3)
points(0.75, 0, pch = 21, lwd = 3, bg = "white", cex = 3)
text(2, 1, "PN", cex = 2)
text(2, 0.5, "EPT", cex = 2)
text(2, 0, "Tracing", cex = 2)

dev.off()



####################################################################
# net monetary benefit

mydata = lapply(c("random", "community", "powerLaw"), function(g){
  nullData = read.csv(paste0("results/high notification/", g,"DF.csv"))
  nullData = subset(nullData, strategy == "null", select = c("strategy", "TotalCost", "Util"))
  nullData = aggregate(cbind(TotalCost, Util)~strategy, nullData, mean)
  nullData$strategy = "NULL"
  nullData$graph = g
  nullData$pr = NA
  nullData$Util = nullData$Util/12
  pr = round(seq(0, 1, 0.01), 2)
  
  out = nullData
  for (st in c("PN", "EPT", "Tracing")){
    tempDF = read.csv(paste0("results/high notification/bayes_estimate_",g,"_", st, ".csv"))
    betas = tempDF[tempDF$X %in% c("beta[1]", "beta[2]", "beta0[1]", "beta0[2]"), "mean"]
    TotalCost = betas[3]+betas[1]*pr
    Util = betas[4]+betas[2]*pr
    
    stDF = data.frame(strategy = st, TotalCost = TotalCost, Util = Util, graph = g, 
                      pr = pr)
    out = rbind(out, stDF)
  }
  
  out$graph = g
  return(out)
})

names(mydata) = c("randomDF", "communityDF", "powerLawDF")

NMB_func = function(dat, p_treat_PN, p_treat_EPT, wtp = 100000){
  PN = subset(dat, strategy == "PN" & pr == round(p_treat_PN, 2))
  EPT = subset(dat, strategy == "EPT" & pr == round(p_treat_EPT, 2))
  
  PN$NMB = PN$Util*wtp - PN$TotalCost
  EPT$NMB = EPT$Util*wtp - EPT$TotalCost
  ExpLoss = abs(PN$NMB-EPT$NMB)
  out = c(p_treat_PN, p_treat_EPT, ExpLoss)
  names(out) = c("PN", "EPT", "ExpLoss")
  return(out)
}

wtp = 100000

g = "random"

p_treat_PN = seq(0.01, 1, 0.01)
p_treat_EPT = seq(0.01, 1, 0.01)
z = matrix(NA, nrow = 100, ncol = 100)
colnames(z) = seq(0.01, 1, 0.01)
rownames(z) = seq(0.01, 1, 0.01)
for(i in seq(0.01, 1, 0.01)) {
  for(j in seq(0.01, 1, 0.01)) {
    out = NMB_func(dat = mydata[[paste0(g, "DF")]], p_treat_PN = i, p_treat_EPT = j, wtp = wtp)
    z[paste0(i), paste0(j)] = -out[['ExpLoss']]
  }
}

write.csv(z, paste0("/Users/szu-yukao/Documents/Network_structure_and_STI/writing/", g, ".csv"))

library(plotly)

p <- plot_ly(x = p_treat_PN, y = p_treat_EPT, z = z) %>% 
  add_surface() %>%
  layout(
    title = paste0(g, "DF"),
    scene = list(
      xaxis = list(title = "p_treat_PN"),
      yaxis = list(title = "p_treat_EPT" ),  
      zaxis = list(title = "Expected Loss($)")
    ))



p <- plot_ly(
  x = p_treat_PN, 
  y = p_treat_EPT, 
  z = z, 
  type = "contour"
) %>% layout(
  title = paste0(g, "DF"),
  scene = list(
    xaxis = list(title = "p_treat_PN"),
    yaxis = list(title = "p_treat_EPT" ),  
    zaxis = list(title = "Expected Loss($)")
  ))



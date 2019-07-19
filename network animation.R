library(igraph)
library(grDevices)

makeTransparent<-function(someColor, alpha=100) {
  newColor<-someColor + alpha #I wish
  return(newColor)
}

#load the edges with time stamp
#there are three columns in edges: id1,id2,time
edges <- read.csv("results/power law graph edgelist.csv",header=T, sep = ",")
colnames(edges) = c("ID1", "ID2", "infection", "time")
node = read.csv("results/power law graph node attribute.csv",header=T, sep = ",")
# edges$duration = edges$terminus - edges$onset
# edges$edge.id = c(1:nrow(edges))

tmp.node = subset(node, time == 1)
tmp.el = data.frame(ID1 = numeric(), ID2 = numeric(), infection = numeric(), time = numeric())
# tmp.el = subset(edges, time == 1)
g = graph.data.frame(tmp.el, directed = F, vertices = tmp.node)
# now we need to associate dates with links and I use sample with replace=T
E(g)$width <- 2
E(g)$color <- ifelse(E(g)$infection == 1, 'firebrick', 'gray80')

V(g)$color = ifelse(V(g)$I == 1, 'firebrick', 'gray80')
# season to taste
V(g)$size <- 3
V(g)$label <- ""

par(mfrow=c(1,1),mar=c(0,0,0,0), oma=c(0,0,0,0))

# make a layout and set the x & y attributes of the graph vertices 
l.old <- layout.fruchterman.reingold(g)

png(file="results/example%03d.png", width=1600,height=1200)
for(i in c(1:60)) {
  print(i)
  tmp.node = subset(node, time == i-1)
  tmp.el = subset(edges, time == i)
  tmp.el = tmp.el[order(tmp.el$ID1, tmp.el$ID2, tmp.el$infection), ]
  tmp.el = tmp.el[!duplicated(tmp.el[, c("ID1", "ID2")]), ]
  # tmp.el = subset(edges, time == 1)
  g = graph.data.frame(tmp.el, directed = F, vertices = tmp.node)
  # now we need to associate dates with links and I use sample with replace=T
  E(g)$width <- ifelse(E(g)$infection == 1, 4, 1)
  E(g)$color <- ifelse(E(g)$infection == 1, 'firebrick1', 'gray70')
  
  V(g)$color = ifelse(V(g)$I == 1, adjustcolor( "firebrick1", alpha.f = 0.8), adjustcolor('deepskyblue2', alpha.f = 0.8))
  # season to taste
  V(g)$size <- degree(g)/2
  V(g)$label <- ""
  
  l.new = layout_with_fr(g,niter=10,coords=l.old, 
                              start.temp=0.05,grid="nogrid")
  
  plot(g,layout=l.new, vertex.frame.color = NA, edge.curved = F, arrow.size = 0, arrow.width = 0, arrow.mode = 0)
  # png(file=paste0("results/power law animation", i,".png"), width=1600,height=900)
  l.old = l.new
}
dev.off()



######################################################################
# Animation of the community graph
library(igraph)

#load the edges with time stamp
#there are three columns in edges: id1,id2,time
edges <- read.csv("results/community graph edgelist.csv",header=T, sep = ",")
colnames(edges) = c("ID1", "ID2", "infection", "time")
node = read.csv("results/community graph node attribute.csv",header=T, sep = ",")
# edges$duration = edges$terminus - edges$onset
# edges$edge.id = c(1:nrow(edges))

tmp.node = subset(node, time == 1)
tmp.el = data.frame(ID1 = numeric(), ID2 = numeric(), infection = numeric(), time = numeric())
# tmp.el = subset(edges, time == 1)

annual.edges <- read.csv("results/community graph annual edgelist.csv",header=T, sep = ",")
annual.edges = annual.edges[, c(3, 4)]

g = graph.data.frame(annual.edges, directed = F, vertices = tmp.node)
# now we need to associate dates with links and I use sample with replace=T
E(g)$width <- 2
E(g)$color <- ifelse(E(g)$infection == 1, 'firebrick', 'gray80')

V(g)$color = ifelse(V(g)$I == 1, 'firebrick', 'gray80')
# season to taste
V(g)$size <- 3
V(g)$label <- ""

par(mfrow=c(1,1),mar=c(0,0,0,0), oma=c(0,0,0,0))

# make a layout and set the x & y attributes of the graph vertices 
l.old <- layout.fruchterman.reingold(g)

png(file="results/example%03d.png", width=1600,height=1200)
for(i in c(1:60)) {
  print(i)
  tmp.node = subset(node, time == i-1)
  tmp.el = subset(edges, time == i)
  tmp.el = tmp.el[order(tmp.el$ID1, tmp.el$ID2, tmp.el$infection), ]
  # tmp.el = subset(edges, time == 1)
  g = graph.data.frame(tmp.el, directed = F, vertices = tmp.node)
  # now we need to associate dates with links and I use sample with replace=T
  E(g)$width <- ifelse(E(g)$infection == 1, 4, 1)
  E(g)$color <- ifelse(E(g)$infection == 1, 'firebrick1', 'gray50')
  
  V(g)$color = ifelse(V(g)$I == 1, adjustcolor( "firebrick1", alpha.f = 0.8), adjustcolor('deepskyblue2', alpha.f = 0.8))
  # season to taste
  V(g)$size <- degree(g)/2
  V(g)$label <- ""
  
  l.new = layout_with_fr(g,niter=10,coords=l.old, 
                         start.temp=0.05,grid="nogrid")
  
  plot(g,layout=l.new, vertex.frame.color = NA, edge.curved = F, arrow.size = 0, arrow.width = 0, arrow.mode = 0)
  # png(file=paste0("results/power law animation", i,".png"), width=1600,height=900)
  l.old = l.new
}
dev.off()




#####################################################################
# graph of the last 12 months. 
library(igraph)

graphs = c("random", "community", "power law")

for(x in c(1:length(graphs))) {
  g.type = graphs[x]
  
  #load the edges with time stamp
  #there are three columns in edges: id1,id2,time
  edges <- read.csv(paste0("results/", g.type, " graph edgelist.csv"),header=T, sep = ",")
  colnames(edges) = c("ID1", "ID2", "infection", "time")
  node = read.csv(paste0("results/", g.type, " graph node attribute.csv"),header=T, sep = ",")
  # edges$duration = edges$terminus - edges$onset
  # edges$edge.id = c(1:nrow(edges))
  
  node = subset(node, time >= 49)
  node.agg = aggregate(I ~ ID, node, max)
  node = node.agg
  edges = subset(edges, time >= 49)
  edges = edges[!duplicated(edges[, c("ID1", "ID2")]), ]
  edges = edges[sample(c(1:nrow(edges)), nrow(edges)*0.5), ]
  
  
  annual.edges <- read.csv(paste0("results/", g.type, " graph annual edgelist.csv"),header=T, sep = ",")
  annual.edges = annual.edges[, c("source", "target")]
  g = graph.data.frame(annual.edges, directed = F, vertices = node)
  l <- layout.fruchterman.reingold(g)
  
  g = graph.data.frame(edges, directed = F, vertices = node)
  # now we need to associate dates with links and I use sample with replace=T
  E(g)$width <- 3
  E(g)$color <- adjustcolor('gray50', alpha.f = 1)
  
  V(g)$color = ifelse(V(g) %in% node$ID[node$I == 1], "orangered", "skyblue2")
  V(g)$color = "skyblue2"
  
  # V(g)$color = ifelse(degree(g) >= 50, adjustcolor('springgreen', alpha.f = 0.8), adjustcolor('deepskyblue2', alpha.f = 0.7))
  # season to taste
  V(g)$size <- ifelse(degree(g) >= 50, 10, 3)
  V(g)$size <- 5
  V(g)$label <- ""
  
  assign(paste0('l', x), l)
  assign(paste0("g", x), g)
}




setEPS()
postscript('results/network plots (combined).eps', height = 10, width = 28)
par(mar=c(0,0,0,0), oma=c(0,0,3,0))
layout(matrix(c(1,2,3),byrow = T, ncol=3),heights=c(4))
plot(g1,layout=l1, vertex.frame.color = "black", edge.curved = F, arrow.size = 0, arrow.width = 0, arrow.mode = 0, asp=0, margin=0)
mtext("(A) Random network",side = 3, line = -1, cex = 3)
plot(g2,layout=l2, vertex.frame.color = "black", edge.curved = F, arrow.size = 0, arrow.width = 0, arrow.mode = 0, asp=0, margin=0)
mtext("(B) Community network",side = 3, line = -1, cex = 3)
plot(g3,layout=l3, vertex.frame.color = "black", edge.curved = F, arrow.size = 0, arrow.width = 0, arrow.mode = 0, asp=0, margin=0)
mtext("(C) Scale-free network",side = 3, line = -1, cex = 3)
dev.off()








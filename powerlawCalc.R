library(poweRlaw)

x = c(1:250) # maximum number of partners is 250

estAlpha = function(x, alpha, m) {
  xx = dpldis(x, 1, alpha)
  temp_exp = (xx/sum(xx))%*%t(t(x))
  out = c(alpha, abs(temp_exp-m), temp_exp)
  names(out) = c("alpha", "difference", 'expectation')
  return(out)
}

alpha_seq = seq(1.00001, 2, 0.01)

out = do.call(rbind, lapply(alpha_seq, estAlpha, x = x, m = 4*10))
out = data.frame(out)
min_alpha = out$alpha[out$difference == min(out$difference)]

PowerLawDistribution = dpldis(x, 1, 1.02)
PowerLawDist = (PowerLawDistribution)/sum(PowerLawDistribution)

fileConn<-file("PowerLawDist(40degree).txt")
writeLines(paste0(PowerLawDist, collapse=", "), fileConn)
close(fileConn)



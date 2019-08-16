library(poweRlaw)

x = c(1:250) # maximum number of partners is 250

estAlpha = function(x, alpha, m) {
  xx = dpldis(x, 1, alpha)
  temp_exp = (xx/sum(xx))%*%t(t(x))
  out = c(alpha, abs(temp_exp-m), temp_exp)
  names(out) = c("alpha", "difference", 'expectation')
  return(out)
}

alpha_seq = seq(0.1, 5, 0.01)

out = do.call(rbind, lapply(alpha_seq, estAlpha, x = x, m = 4*5))
out = data.frame(out)
min_alpha = out$alpha[out$difference == min(out$difference)]

PowerLawDistribution = dpldis(x, 1, 1.31)
PowerLawDist = (PowerLawDistribution)/sum(PowerLawDistribution)

fileConn<-file("PowerLawDist(200degree).txt")
writeLines(paste0(PowerLawDist, collapse=", "), fileConn)
close(fileConn)



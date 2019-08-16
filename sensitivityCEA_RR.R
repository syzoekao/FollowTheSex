##############################################
# Estimation (Frequentist)
##############################################

rm(list = ls())
library(data.table)
library(splines)
library(mgcv)
library(dplyr)

predData = function(g, cor_or_not, lvl, add_str) {
  mydata = data.table(read.csv(paste0("results/RRresults3/", g, cor_or_not, lvl, add_str,".csv")), 
                      key = c("strategy"))
  mydata[, 'pr'] = mydata[, 'p_treat_pn'] + mydata[, 'p_treat_ept'] + mydata[, 'p_treat_tr']
  colsel = c("strategy", "tot_person_time", "TotalCost", "pr")
  out = mydata[, ..colsel]
  out[, "TotalCost"] <- out[, "TotalCost"] / 1000
  colnames(out) = c("strategy", "Util", "TotalCost", "pr")
  out = out[strategy != "screen", ]
  
  mean_out <- out %>% group_by(strategy, pr) %>% 
    summarize(mean_C = mean(TotalCost), 
              mean_U = mean(Util))
  
  min_C <- min(mean_out$mean_C)
  max_C <- max(mean_out$mean_C)
  min_U <- min(mean_out$mean_U)
  max_U <- max(mean_out$mean_U)
  
  pred_out <- lapply(c("pn", "ept", "trace"), function(x, graph = g) {
    data.frame(pr = seq(0, 1, 0.01), strategy = x, 
               graph = graph, pred_C = NA, pred_U = NA)
  })
  
  ## Cost
  # PN
  tmp_data <- out[strategy == 'pn', ]
  tmp_mean <- mean_out %>% filter(strategy == 'pn')
  gam_C <- gam(TotalCost ~ s(pr), data = tmp_data)
  pred_out[[1]]$pred_C <- predict(gam_C, newdata = pred_out[[1]])
  plot(tmp_mean$pr, tmp_mean$mean_C, type = "l", xlim = c(0, 1), 
       ylim = c(min_C, max_C), lwd = 1, xlab = "pr", ylab = "cost ($1,000)", 
       main = paste0(g))
  points(tmp_mean$pr, tmp_mean$mean_C, pch = 21, bg = "white", lwd = 1)
  lines(pred_out[[1]]$pr, pred_out[[1]]$pred_C, col = "dodgerblue", lwd = 1)
  
  # EPT
  tmp_data <- out[strategy == 'ept', ]
  tmp_mean <- mean_out %>% filter(strategy == 'ept')
  gam_C <- gam(TotalCost ~ s(pr), data = tmp_data)
  # gam.check(gam_C)
  pred_out[[2]]$pred_C <- predict(gam_C, newdata = pred_out[[2]])
  lines(tmp_mean$pr, tmp_mean$mean_C, lwd = 1)
  points(tmp_mean$pr, tmp_mean$mean_C, pch = 22, bg = "white", lwd = 1)
  lines(pred_out[[2]]$pr, pred_out[[2]]$pred_C, col = "limegreen", lwd = 1)
  
  # Tracing
  tmp_data <- out[strategy == 'trace', ]
  tmp_mean <- mean_out %>% filter(strategy == 'trace')
  gam_C <- gam(TotalCost ~ s(pr), data = tmp_data)
  # gam.check(gam_C)
  pred_out[[3]]$pred_C <- predict(gam_C, newdata = pred_out[[3]])
  lines(tmp_mean$pr, tmp_mean$mean_C, lwd = 1)
  points(tmp_mean$pr, tmp_mean$mean_C, pch = 23, bg = "white", lwd = 1)
  lines(pred_out[[3]]$pr, pred_out[[3]]$pred_C, col = "salmon", lwd = 1)
  
  legend("topleft", pch = c(21, 22, 23), 
         bg = c("white", "white", "white"), 
         lty = c(1, 1, 1), 
         col = c("dodgerblue", "limegreen", "salmon"), 
         legend = c("PN", "EPT", "Tracing"))

  ## Person time
  # PN
  tmp_data <- out[strategy == 'pn', ]
  tmp_mean <- mean_out %>% filter(strategy == 'pn')
  gam_U <- gam(Util ~ s(pr), data = tmp_data)
  pred_out[[1]]$pred_U <- predict(gam_U, newdata = pred_out[[1]])
  plot(tmp_mean$pr, tmp_mean$mean_U, type = "l", xlim = c(0, 1), 
       ylim = c(min_U, max_U), lwd = 1, xlab = "pr", ylab = "person years", 
       main = paste0(g))
  points(tmp_mean$pr, tmp_mean$mean_U, pch = 21, bg = "white", lwd = 1)
  lines(pred_out[[1]]$pr, pred_out[[1]]$pred_U, col = "dodgerblue", lwd = 1)
  
  # EPT
  tmp_data <- out[strategy == 'ept', ]
  tmp_mean <- mean_out %>% filter(strategy == 'ept')
  gam_U <- gam(Util ~ s(pr), data = tmp_data)
  # gam.check(gam_U)
  pred_out[[2]]$pred_U <- predict(gam_U, newdata = pred_out[[2]])
  lines(tmp_mean$pr, tmp_mean$mean_U, lwd = 1)
  points(tmp_mean$pr, tmp_mean$mean_U, pch = 22, bg = "white", lwd = 1)
  lines(pred_out[[2]]$pr, pred_out[[2]]$pred_U, col = "limegreen", lwd = 1)
  
  # Tracing
  tmp_data <- out[strategy == 'trace', ]
  tmp_mean <- mean_out %>% filter(strategy == 'trace')
  gam_U <- gam(Util ~ s(pr), data = tmp_data)
  # gam.check(gam_U)
  pred_out[[3]]$pred_U <- predict(gam_U, newdata = pred_out[[3]])
  lines(tmp_mean$pr, tmp_mean$mean_U, lwd = 1)
  points(tmp_mean$pr, tmp_mean$mean_U, pch = 23, bg = "white", lwd = 1)
  lines(pred_out[[3]]$pr, pred_out[[3]]$pred_U, col = "salmon", lwd = 1)
  
  legend("topright", pch = c(21, 22, 23), 
         bg = c("white", "white", "white"), 
         lty = c(1, 1, 1), 
         col = c("dodgerblue", "limegreen", "salmon"), 
         legend = c("PN", "EPT", "Tracing"))
  
  pred_out <- do.call(rbind, pred_out)
  pred_out$pred_C <- pred_out$pred_C * 1000
  return(pred_out)
}



for (add_str in c("(corr_scr 2yrs)")) {#, , "(uncorr_scr 2yrs)", "(corr_scr 2yrs)", "(uncorr_scr 2yrs)")) {
  for (cor_or_not in c("Corr")) { # "Uncorr", 
    for (lvl in c("Low", "High")) {
      setEPS()
      postscript(paste0('/Users/szu-yukao/Documents/Network_structure_and_STI/networkSTI/results/RRsummary3/CE comparison ', 
                        cor_or_not, lvl, add_str, '.eps'), height = 12, width = 10)
      # par(mar=c(6, 8, 5, 3), oma = c(0, 0, 0, 0)) 
      layout(matrix(c(1,2,3,4,5,6,7,8), byrow = T, ncol=2),heights=c(5, 5, 5, 5))
      
      randomDF <- predData(g = "random", cor_or_not, lvl, add_str)
      communityDF <- predData(g = "community", cor_or_not, lvl, add_str)
      power_lawDF <- predData(g = "power_law", cor_or_not, lvl, add_str)
      empiricalDF <- predData(g = "empirical", cor_or_not, lvl, add_str)
      
      dev.off()
      
      gam_out <- rbind(randomDF, communityDF, power_lawDF, empiricalDF)
      write.csv(gam_out, paste0("results/RRresults3/", cor_or_not, lvl, add_str, "_estimate.csv"), row.names = F)
    }
  }
}



###########################################
# smooth function
###########################################
library(ggplot2)
library(reshape2)
library(ggpubr)


CE.function = function(data, graph, p_treat_pn, p_treat_ept, wtp = 100000) {
  data = data[data$graph == graph, ]
  null.set = data[data$strategy == "screen", ]
  tmp.PN = data[(data$strategy == "pn") & (data$pr == round(p_treat_pn,2)), ]
  # tmp.Tr = data[(data$strategy == "Tracing") & (data$pr == round(max(c(p_treat_PN, p_treat_EPT)),2)), ]
  tmp.Tr = data[(data$strategy == "trace") & (data$pr == round(max(c(p_treat_pn, p_treat_ept)),2)), ]
  tmp.EPT = data[(data$strategy == "ept") & (data$pr == round(p_treat_ept,2)), ]
  tmp.dat = rbind(null.set, tmp.EPT, tmp.PN, tmp.Tr)
  tmp.dat = data.frame(tmp.dat)
  tmp.dat = tmp.dat[order(tmp.dat$pred_C), ]
  tmp.dat$pred_U = -tmp.dat$pred_U
  
  C.matrix = matrix(rep(tmp.dat$pred_C, each = 4), nrow = 4, byrow = T) - matrix(rep(tmp.dat$pred_C, 4), nrow = 4, byrow = T)
  U.matrix = matrix(rep(tmp.dat$pred_U, each = 4), nrow = 4, byrow = T) - matrix(rep(tmp.dat$pred_U, 4), nrow = 4, byrow = T)
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
    opt.st = "screen"
  } else {
    opt.st = opt.st
  }
  
  return(list(ce.df = ce.df, opt = opt.st))
}

get_CEA_matrix <- function(g, df, wtp) {
  
  tmp_mat = matrix(NA, nrow = 100, ncol = 100)
  colnames(tmp_mat) = seq(0.01, 1, 0.01)
  rownames(tmp_mat) = rev(seq(0.01, 1, 0.01))
  for(j in seq(0.01, 1, 0.01)) {
    for(i in seq(0.01, 1, 0.01)) {
      out = CE.function(data = df, graph = g, p_treat_pn = j, p_treat_ept = i, wtp = wtp)
      tmp_mat[paste0(j), paste0(i)] = as.character(out[["opt"]])
    }
  }
  
  out = data.frame(tmp_mat)
  out$p_treat_pn <- rownames(out)
  out$p_treat_pn <- as.numeric(out$p_treat_pn)
  out <- melt(out, id.vars = "p_treat_pn")
  colnames(out)[2] <- "p_treat_ept"
  out$p_treat_ept <- as.character(out$p_treat_ept)
  out$p_treat_ept <- as.numeric(gsub("X", "", out$p_treat_ept))
  out$strategy <- factor(out$value, levels = c("screen", "pn", "ept", "trace"))
  return(out)
}




for (add_str in c("(corr_scr 2yrs)")) { # , "(uncorr_scr 2yrs)"
  for (cor_or_not in c("Corr")) { # "Uncorr", 
    for (lvl in c("Low", "High")) {
      mydata = lapply(c("random", "community", "power_law", "empirical"), 
                    function(g){
                      nullData = read.csv(paste0("results/RRresults3/", 
                                                 g, cor_or_not, 
                                                 lvl, add_str, ".csv"))
                      nullData = subset(nullData, strategy == "screen", 
                                        select = c("strategy", "TotalCost", "tot_person_time"))
                      nullData$tot_person_time = nullData$tot_person_time
                      colnames(nullData) = c("strategy", "pred_C", "pred_U")
                      nullData = aggregate(cbind(pred_C, pred_U) ~ strategy, nullData, mean)
                      nullData$strategy = "screen"
                      nullData$graph = g
                      nullData$pr = NA
                      return(nullData)
                    })
    
      mydata <- do.call(rbind, mydata)
      tempDF = read.csv(paste0("results/RRresults3/", cor_or_not, lvl, add_str, "_estimate.csv"))
      tempDF <- rbind(tempDF, mydata)
      
      wtp = 200
      
      mat_ls <- lapply(c("random", "community", "power_law", "empirical"), 
                       get_CEA_matrix, df = tempDF, wtp = wtp)
      
      p1 <- ggplot(mat_ls[[1]], aes(x = p_treat_ept, y = p_treat_pn, fill = strategy)) + 
        geom_tile() +
        scale_fill_manual(values=c("gray80", "yellowgreen", "cornflowerblue", "gold"), 
                          labels = c("screening alone", "PN", "EPT", "Tracing"), 
                          drop = F) + 
        geom_point(x = 0.79, y = 0.71, pch = "*", cex = 12, colour = "firebrick", show.legend = FALSE) + 
        ggtitle("(A) Random network") + 
        xlab("EPT partner compliance") +
        ylab("PN partner compliance") + 
        theme_bw() + 
        theme(panel.grid.major = element_blank(),
              panel.grid.minor = element_blank(),
              plot.title = element_text(hjust = 0.5)) 
        
      p2 <- ggplot(mat_ls[[2]], aes(x = p_treat_ept, y = p_treat_pn, fill = strategy)) + 
        geom_tile() +
        scale_fill_manual(values=c("gray80", "yellowgreen", "cornflowerblue", "gold"), 
                          labels = c("screening alone", "PN", "EPT", "Tracing"), 
                          drop = F) + 
        geom_point(x = 0.79, y = 0.71, pch = "*", cex = 12, colour = "firebrick", show.legend = FALSE) + 
        ggtitle("(B) Community network") + 
        xlab("EPT partner compliance") +
        ylab("PN partner compliance") + 
        theme_bw() + 
        theme(panel.grid.major = element_blank(),
              panel.grid.minor = element_blank(),
              plot.title = element_text(hjust = 0.5)) 
      
      p3 <- ggplot(mat_ls[[3]], aes(x = p_treat_ept, y = p_treat_pn, fill = strategy)) + 
        geom_tile() +
        scale_fill_manual(values=c("gray80", "yellowgreen", "cornflowerblue", "gold"), 
                          labels = c("screening alone", "PN", "EPT", "Tracing"), 
                          drop = F) + 
        ggtitle("(C) Scale-free network") + 
        geom_point(x = 0.79, y = 0.71, pch = "*", cex = 12, colour = "firebrick", show.legend = FALSE) + 
        xlab("EPT partner compliance") +
        ylab("PN partner compliance") + 
        theme_bw() + 
        theme(panel.grid.major = element_blank(),
              panel.grid.minor = element_blank(),
              plot.title = element_text(hjust = 0.5)) 
      
      p4 <- ggplot(mat_ls[[4]], aes(x = p_treat_ept, y = p_treat_pn, fill = strategy)) + 
        geom_tile() +
        scale_fill_manual(values=c("gray80", "yellowgreen", "cornflowerblue", "gold"), 
                          labels = c("screening alone", "PN", "EPT", "Tracing"), 
                          drop = F) + 
        ggtitle("(D) Empirical network") + 
        geom_point(x = 0.79, y = 0.71, pch = "*", cex = 12, colour = "firebrick", show.legend = FALSE) + 
        xlab("EPT partner compliance") +
        ylab("PN partner compliance") + 
        theme_bw() + 
        theme(panel.grid.major = element_blank(),
              panel.grid.minor = element_blank(),
              plot.title = element_text(hjust = 0.5)) 
      
      all_p <- ggarrange(p1, p2, p3, p4, ncol=2, nrow=2, common.legend = TRUE, legend="bottom")
      ggsave(paste0("results/RRsummary3/", cor_or_not, lvl, add_str, "_CEmatrix", wtp, ".png"), 
             plot = all_p)
    }
  }
}

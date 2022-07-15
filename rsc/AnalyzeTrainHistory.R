packages <- c("ggplot2", "reshape2", "dplyr")
for(i in 1:NROW(packages)){
  if(!require(packages[i], character.only = TRUE)){
    install.packages(packages[i])
    library(packages[i], character.only = TRUE)
  }
}

model <- "U256"

dir_parent <- "/home/manuel/Dropbox/tmp"
dir_sub <- "cpts/trained_mod"
filename <- "history.csv"

cols <- c(
  rgb(0, 150, 130, alpha = 254.99, maxColorValue = 255), #kit colour
  rgb(70, 100, 170, alpha = 254.99, maxColorValue = 255), #kit blue
  rgb(223, 155, 27, alpha = 254.99, maxColorValue = 255), #kit orange
  rgb(140, 182, 60, alpha = 254.99, maxColorValue = 255), #kit Mai green
  rgb(162, 34, 35, alpha = 254.99, maxColorValue = 255), #kit red
  rgb(163, 16, 124, alpha = 254.99, maxColorValue = 255), #kit violet
  rgb(167, 130, 46, alpha = 254.99, maxColorValue = 255), #kit brown
  rgb(252, 229, 0, alpha = 254.99, maxColorValue = 255), #kit yellow
  rgb(25, 161, 224, alpha = 254.99, maxColorValue = 255)# kit cyan
)

hist <- read.csv(file.path(dir_parent, model, dir_sub, filename))
string <- "tf.Tensor(287.21475315093994, shape=(), dtype=float64)"
extr_secs <- function(string){
  t_chr <- substring(string, regexpr("Tensor", string) + 7, regexpr(", ", string) - 1)
  t_num <- as.numeric(t_chr)
  return(t_num)
}

hist$t_sec <- extr_secs(hist$EpochTime)

t_mean <- mean(hist$t_sec)
t_mean
t_sd <- sd(hist$t_sec)
t_sd

require("ggplot2")
require("reshape2")
require("dplyr")
history <- melt(hist, id.vars = "X")
history$value <- as.numeric(history$value)

vars = c("val_loss", "val_Sparse_MeanIoU", "val_sparse_categorical_accuracy")
hist_subset <- history[history$variable %in% vars,]
maxima = list()
minima = list()
for(v in unique(hist_subset$variable)){
  vals <- hist_subset[which(hist_subset$variable == v),]
  EpochAtMinVal <- vals$X[which(vals$value == min(vals$value))]
  minima[v] <- EpochAtMinVal
  EpochAtMaxVal <- vals$X[which(vals$value == max(vals$value))]
  maxima[v] <- EpochAtMaxVal
}

gg <- ggplot(data = hist_subset, aes(x = X, y = value)) +
  geom_line(aes(colour = variable)) +
  scale_color_manual(values = cols)

for(i in 1:length(maxima)){
  variable <- names(maxima)[i]
  value <- maxima[[i]]
  gg <- gg +
    geom_vline(xintercept = value, colour = cols[i], linetype = 3) +
    xlab("Epoch")
}

for(i in 1:length(minima)){
  variable <- names(minima)[i]
  value <- minima[[i]]
  gg <- gg +
    geom_vline(xintercept = value, colour = cols[i], linetype = 4)
}

png(file.path(dir_parent, "Model_hist.png"), width = 512, height = 256)
plot(gg)
dev.off()

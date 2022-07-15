wd <- "C:/Users/Manuel/Nextcloud/Masterarbeit/out/mod_FCD"
dat <- read.table(file.path(wd, "confusion_matr.csv"), header = FALSE, sep = "\t")
colnames(dat) <- seq(1, ncol(dat))
rownames(dat) <- seq(1, ncol(dat))

library(tidyverse)
data <- dat %>%
  rownames_to_column() %>%
  gather(colname, value, -rowname)
data$rowname <- as.ordered(as.numeric(data$rowname))
data$colname <- as.ordered(as.numeric(data$colname))
data$value <- as.ordered(data$value)
data$value <- as.numeric(data$value)
ggplot(data, aes(x = rowname, y = colname, fill = log(value + 0.01, base = 10))) +
  geom_tile() +
  scale_fill_continuous(type = "viridis")
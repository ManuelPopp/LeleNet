#############################################
##### Load packages, set theme
#############################################
packages <- c("readxl", "ggplot2", "dplyr", "reshape2")
for(i in 1:NROW(packages)){
  if(!require(packages[i], character.only = TRUE)){
    install.packages(packages[i])
    library(packages[i], character.only = TRUE)
  }
}

# style options
theme_set(theme_bw())
cols <- c(
  rgb(0, 150, 130, alpha = 254.99, maxColorValue = 255), #kit colour
  rgb(70, 100, 170, alpha = 254.99, maxColorValue = 255), #kit blue
  rgb(223, 155, 27, alpha = 254.99, maxColorValue = 255), #kit orange
  rgb(140, 182, 60, alpha = 254.99, maxColorValue = 255), #kit Mai green
  rgb(162, 34, 35, alpha = 254.99, maxColorValue = 255), #kit red
  rgb(163, 16, 124, alpha = 254.99, maxColorValue = 255), #kit violet
  rgb(167, 130, 46, alpha = 254.99, maxColorValue = 255), #kit brown
  rgb(252, 229, 0, alpha = 254.99, maxColorValue = 255), #kit yellow
  rgb(25, 161, 224, alpha = 254.99, maxColorValue = 255) # kit cyan
)
# graphics width mm for graphics device input (expects inch)
w_90mm <- 3.54331
# two-column graphics
w <- 2 * w_90mm
h <- w_90mm * 3 / 4

#############################################
##### Load data
#############################################
dir_dropbox <- "C:/Users/Manuel/Dropbox/Apps/Overleaf/Masterarbeit/fig"
excel_path <- "D:/Dateien/Studium_KIT/Master_GOEK/Masterarbeit/dat/xls/"
dat <- as.data.frame(readxl::read_excel(file.path(excel_path, "Accuracy_table.xlsx"),
                                        sheet = "Complete", skip = 1))[, c(1:12)]
names(dat) <- c("Class", paste0(rep(c("U", "F", "D"), each = 3), rep(c(256, 512, 1024), 3)),
                "a512", "b512")
dat[dat == "-"] <- NA
dat <- dat[-which(dat$Class == "Mundulea sericea"),]
for(j in 2:ncol(dat)){
  dat[, j] <- as.numeric(dat[, j])
}

dat_l <- reshape2::melt(dat[seq(1, 16), seq(1, 10)], id.var = "Class")
dat_l$Tilesize <- as.numeric(sub("\\w", "", dat_l$variable))
dat_l$CNN <- substr(dat_l$variable, 1, 1)
dat_l$CNN[dat_l$CNN %in% c("a", "b")] <- "Within block"
dat_l$CNN[dat_l$CNN == "U"] <- "U-Net"
dat_l$CNN[dat_l$CNN == "F"] <- "FC-DenseNet"
dat_l$CNN[dat_l$CNN == "D"] <- "DeepLabv3+"
dat_l$CNN <- factor(dat_l$CNN, levels = c("U-Net", "FC-DenseNet", "DeepLabv3+", "Within block"))
dat_l$Tilesize <- factor(dat_l$Tilesize, levels = c("256", "512", "1024"))
dat_l$boxes <- as.character(dat_l$Tilesize)
dat_l$boxes[dat_l$variable == "a512"] <- "a"
dat_l$boxes[dat_l$variable == "b512"] <- "b"
dat_l$boxes <- factor(dat_l$boxes, levels = c("256", "512", "1024", "a", "b"))

#############################################
##### two factor repeated measures ANOVA
#############################################
anova <- aov(value ~ CNN * Tilesize + Error(Class/(CNN*Tilesize)), data = dat_l)
summary(anova)

#############################################
##### t-tests
#############################################
models <- names(dat)[-c(1, 11, 12)]
t_tests_p <- matrix(nrow = length(models) - 1, ncol = length(models) - 1)
t_tests_t <- matrix(nrow = length(models) - 1, ncol = length(models) - 1)

row.names(t_tests_p) <- models[-1]
colnames(t_tests_p) <- models[-9]
row.names(t_tests_t) <- models[-1]
colnames(t_tests_t) <- models[-9]

for(i in 1:(nrow(t_tests) - 1)){
  for(j in 1:(ncol(t_tests) - 1)){
    t_test <- t.test(dat[, which(names(dat) == row.names(t_tests_p)[i])],
                            dat[, which(names(dat) == colnames(t_tests_p)[j])],
                            paired = TRUE)
    if(i >= j){
      t_tests_p[i, j] <- signif(t_test$p.value, 3)
      t_tests_t[i, j] <- signif(t_test$statistic, 3)
    }else{
      t_tests_p[i, j] <- NA
      t_tests_t[i, j] <- NA
    }
  }
}

write.table(t_tests_p, file.path(excel_path, "Paired_t_tests_p.csv"), sep = " & ", eol = "\\\\\n")
write.table(t_tests_t, file.path(excel_path, "Paired_t_tests_t.csv"), sep = " & ", eol = "\\\\\n")

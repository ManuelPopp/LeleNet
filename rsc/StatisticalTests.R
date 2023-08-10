#############################################
##### Load packages, set theme
#############################################
if(!require("BiocManager", character.only = TRUE)){
  install.packages("BiocManager")
}

# in case the following fails due to issues with nloptr,
# run in terminal: sudo apt-get install libnlopt-dev
packages <- c("lme4", "lmerTest", "emmeans", "pbkrtest")
for(i in 1:NROW(packages)){
  if(!require(packages[i], character.only = TRUE)){
    BiocManager::install(packages[i])
    library(packages[i], character.only = TRUE)
  }
}

packages <- c("readxl", "dplyr", "reshape2", "ggplot2", "raster", "ggplot2", "landscapemetrics")
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

# Remove Mundulea sericea, since the species was not found in any of the training plots
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
##### Pairwise comparison
#############################################
models <- names(dat)[-c(1, 11, 12)]
t_tests_p <- matrix(nrow = length(models) - 1, ncol = length(models) - 1)
t_tests_t <- matrix(nrow = length(models) - 1, ncol = length(models) - 1)

row.names(t_tests_p) <- models[-1]
colnames(t_tests_p) <- models[-9]
row.names(t_tests_t) <- models[-1]
colnames(t_tests_t) <- models[-9]

shapiro_mat <- t_tests_p

for(i in 1:(nrow(t_tests_p))){
    for(j in 1:(ncol(t_tests_p))){
        t_test <- t.test(dat[, which(names(dat) == row.names(t_tests_p)[i])],
                         dat[, which(names(dat) == colnames(t_tests_p)[j])],
                         paired = TRUE)
        
        Difference = dat[, which(names(dat) == row.names(t_tests_p)[i])] -
            dat[, which(names(dat) == colnames(t_tests_p)[j])]
        
        hist(Difference,   
             col = "gray", 
             main = "Histogram of differences",
             xlab = "Difference")
        
        if(i >= j){
            if(t_test$p.value < 0.001){
                t_tests_p[i, j] <- "\\textless 0.001"
            }else if(t_test$p.value < 0.01){
                t_tests_p[i, j] <- as.character(round(t_test$p.value, 3))
            }else{
                t_tests_p[i, j] <- as.character(round(t_test$p.value, 2))
            }
            
            t_tests_t[i, j] <- as.character(signif(t_test$statistic, 3))
            
            shapwilk <- shapiro.test(Difference)
            shapiro_mat[i, j] <- round(shapwilk$p.value, 2)
            
        }else{
            t_tests_p[i, j] <- " "
            t_tests_t[i, j] <- " "
            shapiro_mat[i, j] <- NA
        }
    }
}

# Gray out all instances where differences were not normally distributed according to the Shapiro-Wilk-Test
t_tests_p[which(shapiro_mat < 0.05)] <- paste0(
    "\\color[gray]{0.5} ",
    t_tests_p[which(shapiro_mat < 0.05)]
    )

t_tests_t[which(shapiro_mat < 0.05)] <- paste0(
    "\\color[gray]{0.5} ",
    t_tests_t[which(shapiro_mat < 0.05)]
)

w_tests_p <- matrix(nrow = length(models) - 1, ncol = length(models) - 1)
w_tests_w <- matrix(nrow = length(models) - 1, ncol = length(models) - 1)

row.names(w_tests_p) <- models[-1]
colnames(w_tests_p) <- models[-9]
row.names(w_tests_w) <- models[-1]
colnames(w_tests_w) <- models[-9]

for(i in 1:(nrow(w_tests_p))){
    for(j in 1:(ncol(w_tests_p))){
        if(i >= j){
            w_test <- wilcox.test(dat[, which(names(dat) == row.names(w_tests_p)[i])],
                                  dat[, which(names(dat) == colnames(w_tests_p)[j])],
                                  paired = TRUE, alternative = "two.sided")
            
            if(w_test$p.value < 0.001){
                w_tests_p[i, j] <- "\\textless 0.001"
            }else if(w_test$p.value < 0.01){
                w_tests_p[i, j] <- as.character(round(w_test$p.value, 3))
            }else{
                w_tests_p[i, j] <- as.character(round(w_test$p.value, 2))
            }
            w_tests_w[i, j] <- as.character(signif(w_test$statistic, 3))
            
        }else{
            w_tests_p[i, j] <- " "
            w_tests_w[i, j] <- " "
        }
    }
}

t_tests_p <- cbind(c(512, 1024, 256, 512, 1024, 256, 512, 1024), t_tests_p)
colnames(t_tests_p)[1] <- "Tile size"

t_tests_t <- cbind(c(512, 1024, 256, 512, 1024, 256, 512, 1024), t_tests_t)
colnames(t_tests_t)[1] <- "Tile size"

w_tests_p <- cbind(c(512, 1024, 256, 512, 1024, 256, 512, 1024), w_tests_p)
colnames(w_tests_p)[1] <- "Tile size"

w_tests_w <- cbind(c(512, 1024, 256, 512, 1024, 256, 512, 1024), w_tests_w)
colnames(w_tests_w)[1] <- "Tile size"

# Export to LaTeX-like tables
write.table(t_tests_p, file.path(excel_path, "Paired_t_tests_p.csv"), sep = " & ", eol = "\\\\\n",
            row.names = TRUE, quote = FALSE
            )

write.table(t_tests_t, file.path(excel_path, "Paired_t_tests_t.csv"), sep = " & ", eol = "\\\\\n",
            row.names = TRUE, quote = FALSE
)

write.table(w_tests_p, file.path(excel_path, "Wilcox_tests_p.csv"), sep = " & ", eol = "\\\\\n",
            row.names = TRUE, quote = FALSE
)

write.table(w_tests_w, file.path(excel_path, "Wilcox_tests_w.csv"), sep = " & ", eol = "\\\\\n",
            row.names = TRUE, quote = FALSE
)

#############################################
##### Load packages, set theme
#############################################
packages <- c("readxl", "ggplot2", "reshape2", "landscapemetrics", "dplyr", "rstatix")
for(i in 1:NROW(packages)){
  if(!require(packages[i], character.only = TRUE)){
    install.packages(packages[i])
    library(packages[i], character.only = TRUE)
  }
}

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
w <- w_90mm
h <- w_90mm * 3 / 4
# two-column graphics
w2 <- 2 * w_90mm

#############################################
##### Load data
#############################################
dir_dropbox <- "C:/Users/Manuel/Dropbox/Apps/Overleaf/Masterarbeit"
excel_path <- "D:/Dateien/Studium_KIT/Master_GOEK/Masterarbeit/dat/xls/"
dat <- as.data.frame(readxl::read_excel(file.path(excel_path, "Accuracy_table.xlsx"),
                                        sheet = "Complete", skip = 1)[, c(1:12)])
names(dat) <- c("Species", paste0(rep(c("U", "F", "D"), each = 3), rep(c(256, 512, 1024), 3)),
                "Same_year", "Subsequent_years")
dat[dat == "-"] <- NA
for(j in 2:ncol(dat)){
  dat[, j] <- as.numeric(dat[, j])
}
datasets <- as.data.frame(readxl::read_excel(file.path(excel_path, "Datasets.xlsx"),
                                              sheet = "Orthomosaics"))

#############################################
##### Calculate landscape metrics
#############################################
r_path <- "D:/Dateien/Studium_KIT/Master_GOEK/Masterarbeit/dat/var"
omk_path <- "G:/Masterarbeit/dat/omk/03_2021"
dir_dropbox <- "C:/Users/Manuel/Dropbox/Apps/Overleaf/Masterarbeit"

finished <- file.exists(file.path(r_path, "lsm_df.Rdata"))
if(!finished){
  train_plots <- datasets$Name[which(datasets$Usage == "Train")]
  for(i in 1:length(train_plots)){
    plot = train_plots[i]
    path <- file.path(omk_path, paste0(plot, "_MASK.tif"))
    omk <- raster::raster(path)
    res <- raster::res(omk)[1]
    e <- raster::extent(omk)
    e[4] <- e[3] + nrow(omk) * res
    x <- raster::raster(ncol = ncol(omk), nrow = nrow(omk), ext = e)
    raster::projection(x) <- "+proj=utm +zone=35 +south +datum=WGS84 +units=m +no_defs"
    omk <- raster::projectRaster(omk, x, method = "ngb")
    patch_areas <- as.data.frame(landscapemetrics::lsm_c_area_mn(omk))[, c(3, 6)]
    class_areas <- as.data.frame(landscapemetrics::lsm_c_ca(omk))[, c(3, 6)]
    # CIRCLE describes the ratio between the patch area and the smallest circumscribing
    # circle of the patch and characterises the compactness of the patch
    circle_mean <- as.data.frame(landscapemetrics::lsm_c_circle_mn(omk))[, c(3, 6)]
    lsm <- data.frame(class = patch_areas[, 1], pa = patch_areas[, 2],
                     ca = class_areas[, 2], cm = circle_mean[, 2],
                     omk = rep(plot, nrow(patch_areas)))
    if(!exists("lsm_df")){
      lsm_df <- lsm
    }else{
      lsm_df <- rbind(lsm_df, lsm)
    }
  }
  summary_lsm <- lsm_df %>%
    group_by(factor(class)) %>%
    summarise(mean_pa = mean(pa), total_ca = sum(ca), mean_cm = mean(cm))
  save(lsm_df, file = file.path(r_path, "lsm_df.Rdata"))
}else{
  load(file.path(r_path, "lsm_df.Rdata"))
  summary_lsm <- lsm_df %>%
    group_by(factor(class)) %>%
    summarise(mean_pa = mean(pa), total_ca = sum(ca), mean_cm = mean(cm))
}

#########################################################################
# same for test data
finished <- file.exists(file.path(r_path, "lsm_df_tst.Rdata"))
if(!finished){
  train_plots <- datasets$Name[which(datasets$Usage == "Test")]
  for(i in 1:length(train_plots)){
    plot = train_plots[i]
    path <- file.path(omk_path, paste0(plot, "_MASK.tif"))
    omk <- raster::raster(path)
    res <- raster::res(omk)[1]
    e <- raster::extent(omk)
    e[4] <- e[3] + nrow(omk) * res
    x <- raster::raster(ncol = ncol(omk), nrow = nrow(omk), ext = e)
    raster::projection(x) <- "+proj=utm +zone=35 +south +datum=WGS84 +units=m +no_defs"
    omk <- raster::projectRaster(omk, x, method = "ngb")
    patch_areas <- as.data.frame(landscapemetrics::lsm_c_area_mn(omk))[, c(3, 6)]
    class_areas <- as.data.frame(landscapemetrics::lsm_c_ca(omk))[, c(3, 6)]
    # CIRCLE describes the ratio between the patch area and the smallest circumscribing
    # circle of the patch and characterises the compactness of the patch
    circle_mean <- as.data.frame(landscapemetrics::lsm_c_circle_mn(omk))[, c(3, 6)]
    lsm <- data.frame(class = patch_areas[, 1], pa = patch_areas[, 2],
                      ca = class_areas[, 2], cm = circle_mean[, 2],
                      omk = rep(plot, nrow(patch_areas)))
    if(!exists("lsm_df_tst")){
      lsm_df_tst <- lsm
    }else{
      lsm_df_tst <- rbind(lsm_df_tst, lsm)
    }
  }
  summary_lsm_tst <- lsm_df %>%
    group_by(factor(class)) %>%
    summarise(mean_pa = mean(pa), total_ca = sum(ca), mean_cm = mean(cm))
  save(lsm_df_tst, file = file.path(r_path, "lsm_df_tst.Rdata"))
}else{
  load(file.path(r_path, "lsm_df_tst.Rdata"))
  summary_lsm_tst <- lsm_df_tst %>%
    group_by(factor(class)) %>%
    summarise(mean_pa = mean(pa), total_ca = sum(ca), mean_cm = mean(cm))
}
names(summary_lsm_tst) <- c("Class", "mean_pa_tst", "total_ca_tst", "mean_cm_tst")
#########################################################################

class_range <- c(1:16)
hist(summary_lsm$mean_pa[class_range])
hist(summary_lsm$total_ca[class_range])
hist(summary_lsm$mean_cm[class_range])

plot(dat$D512[class_range] ~ summary_lsm$mean_pa[class_range])
lin_mod <- lm(dat$D512[class_range] ~ summary_lsm$mean_pa[class_range])
#plot(lin_mod)
summary(lin_mod)

plot(dat$D512[class_range] ~ summary_lsm$total_ca[class_range])
lin_mod <- lm(dat$D512[class_range] ~ summary_lsm$total_ca[class_range])
#plot(lin_mod)
summary(lin_mod)

plot(dat$D512[class_range] ~ summary_lsm$mean_cm[class_range])
lin_mod <- lm(dat$D512[class_range] ~ summary_lsm$mean_cm[class_range])
#plot(lin_mod)
summary(lin_mod)

plot(dat$D512[class_range] ~ summary_lsm$mean_cm[class_range])
lin_mod <- lm(dat$D512[class_range] ~ summary_lsm$mean_pa[class_range] *
                summary_lsm$mean_cm[class_range])
#plot(lin_mod)
summary(lin_mod)

d <- dat[class_range,]
l <- cbind(summary_lsm[class_range,], summary_lsm_tst[class_range,-1])
correlations_p <- data.frame(Model = names(d[, -1]))
correlations_b <- data.frame(Model = names(d[, -1]))
for(metric in names(l[, -1])){
  cor_p <- c()
  cor_b <- c()
  for(model in names(d[, -1])){
    F1 <- d[, model]
    lmetr <- l[, metric]
    lin_cor <- lm(F1 ~ lmetr)
    p <- round(summary(lin_cor)$coefficients[2, 4], 2)
    plot(F1 ~ lmetr, main = paste0(model, ", ", metric, ", p = ", p))
    abline(a = lin_cor$coefficients[1], b = lin_cor$coefficients[2], col = "red")
    summary(lin_cor)
    cor_p <- c(cor_p, p)
    cor_b <- c(cor_b, lin_cor$coefficients[2])
  }
  correlations_p[, metric] <- cor_p
  correlations_b[, metric] <- cor_b
}
for(j in 2:ncol(correlations_b)){
  correlations_b[, j] <- signif(correlations_b[, j], 3)
  correlations_b[, j] <- as.character(correlations_b[, j])
  for(i in 1:nrow(correlations_b)){
    if(correlations_p[i, j] < 0.05){
      correlations_b[i, j] <- paste0(correlations_b[i, j], "\\(^\\ast\\)")
    }
    if(correlations_p[i, j] < 0.01){
      correlations_b[i, j] <- paste0(correlations_b[i, j], "\\(^\\ast\\)")
    }
  }
}

write.table(correlations_b, file = file.path(excel_path, "LSM_Impact.tex"),
           col.names = c("{Model}", "{Mean patch area}", "{Class area}", "{Compactness}",
                         "{Mean patch area}", "{Class area}", "{Compactness}"),
           row.names = FALSE, sep = " & ", eol = "\\\\\n", quote = FALSE)

# ANOVA
dl <- melt(dat, id.vars = "Species")
dl$value <- as.numeric(dl$value)
dl$Network <- factor(c("U-Net", "FC-DenseNet",
                "DeepLabv3+", "Other")[as.numeric(
                  factor(
                    substr(
                      dl$variable, 1, 1), levels = c("U", "F", "D", "S"))
                  )], levels = c("U-Net", "FC-DenseNet",
                                 "DeepLabv3+", "Other"))
dl$Tilesize <- sub("\\w", "", dl$variable)
longdat <- dl[which(dl$Network %in% c("U-Net", "FC-DenseNet", "DeepLabv3+")),]

ANOVA <- aov(value ~ Network * Tilesize, data = longdat)
summary(ANOVA)
# no interaction
ANOVA <- aov(value ~ Network + Tilesize, data = longdat)
ANOVA_SUMMARY <- summary(ANOVA)[[1]]
AOV_TABLE <- ANOVA_SUMMARY[c(1, 2), c(1, 4, 5)]

test_assumptions <- lm(value ~ Network * Tilesize, data = longdat)
plot(test_assumptions)
shap0 <- rstatix::shapiro_test(residuals(test_assumptions))
if(shap0$p.value > 0.05){
  print("Residuals normally distributed.")
}else{
  print("Assumption of normal distribution violated.")
}

shap1 <- longdat %>%
  group_by(variable) %>%
  summarise(shapiro = rstatix::shapiro_test(value))
normal <- which(shap1$shapiro$p.value > 0.05)
if(length(normal) == length(shap1$shapiro$p.value)){
  print("Values within groups normally distributed.")
}else{
  print("Assumption of normal distribution violated.")
}

lev <- longdat %>% rstatix::levene_test(value ~ variable)
if(lev$p > 0.05){
  print("Variances homogeneous.")
}else{
  print("Assumption of homogeneity of variances violated.")
}

hsd <- longdat %>% rstatix::tukey_hsd(value ~ Network * Tilesize)
hsd

# Write data to LaTeX table
#lines <- "\\begin{table}[htbp]\n\\caption[ANOVA]{ANOVA results}
#\\begin{tabular}[lccc]
#\\toprule
#Variable & df & F-value & \\textit{p}-value\\\\
#\\midrule"
#sink(file.path(dir_dropbox, "tab", "ANOVA.tex"))
#cat(lines)
#sink()
long_ttest <- longdat[(longdat$variable %in% levels(longdat$variable)[1:9]), c("Species", "variable", "value")]
long_ttest <- long_ttest[!is.na(long_ttest$value),]
long_ttest$variable <- factor(long_ttest$variable, levels = unique(long_ttest$variable))

t_test <- long_ttest %>%
  pairwise_t_test(
    value ~ variable, paired = TRUE, 
    p.adjust.method = "bonferroni"
  ) %>%
  select(-df, -statistic, -p)
t_test

t_test_df <- as.data.frame(t_test)[, c(2, 3, 6)]
t_test_matr <- matrix(nrow = length(levels(long_ttest$variable)) - 1,
                      ncol = length(levels(long_ttest$variable)))
row.names(t_test_matr) <- levels(long_ttest$variable)[-1]
colnames(t_test_matr) <- levels(long_ttest$variable)
for(i in 1:nrow(t_test_matr)){
  for(j in 1:ncol(t_test_matr)){
    groups <- c(row.names(t_test_matr)[i], colnames(t_test_matr)[j])
    if(length(unique(groups)) == 2){
      t_test_matr[i, j] <- t_test_df$p.adj[which(t_test_df$group1 %in% groups &
                                                   t_test_df$group2 %in% groups)]
    }
  }
}

write.csv2(t_test_matr, file = file.path(excel_path, "Pairwise_t_tests.csv"))

# output summary lsm
lsm <- summary_lsm
lsm$mean_pa <- summary_lsm$mean_pa / max(summary_lsm$mean_pa)
lsm$total_ca <- summary_lsm$total_ca / max(summary_lsm$total_ca)
lsm$percent_ca <- summary_lsm$total_ca / sum(summary_lsm$total_ca)
lsm$mean_cm <- summary_lsm$mean_cm / max(summary_lsm$mean_cm)
classnames <- c("Other woody vegetation",
  expression(italic("Burkea africana")),
  expression(italic("Combretum apiculatum")),
  expression(italic("Combretum molle")),
  expression(italic("Combretum zeyheri")),
  expression(italic("Commiphora mollis")),
  expression(italic("Dichrostachys cinerea")),
  expression(italic("Diplorhynchus condylocarpon")),
  expression(italic("Elephantorrhiza burkei")),
  expression(italic("Grewia")~"spec."),
  expression(italic("Lannea discolor")),
  expression(italic("Mundulea sericea")),
  expression(italic("Ozoroa paniculosa")),
  expression(italic("Pseudolachnostylis maprouneifolia")),
  expression(italic("Pterocarpus rotundifolius")),
  expression(italic("Terminalia sericea")),
  "Bare ground")
names(lsm)[1] <- "Class"
lsm_l <- rbind(setNames(cbind(lsm[, c(1, 2)], rep("Mean patch area", nrow(lsm))),
                        c("Class", "Value", "Variable")),
               setNames(cbind(lsm[, c(1, 4)], rep("Mean compactness", nrow(lsm))),
                        c("Class", "Value", "Variable")),
               setNames(cbind(lsm[, c(1, 5)], rep("Fractional cover", nrow(lsm))),
                              c("Class", "Value", "Variable")))
gg <- ggplot(data = lsm_l, aes(x = Class, y = Value, colour = Variable, shape = Variable)) +
  geom_point() +
  scale_x_discrete(labels = classnames) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5),
        axis.title.x = element_blank(),
        axis.title.y = element_blank(),
        legend.position = "None") +
  scale_color_manual(name = "Variable", values = cols)

pdf(file.path(dir_dropbox, "fig", "Class_LSM.pdf"), width = w, height = h * 1.5)
gg
dev.off()

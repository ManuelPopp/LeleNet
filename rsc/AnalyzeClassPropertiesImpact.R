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

class_range <- c(2:16)
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
metric_names <- names(summary_lsm)[-1]
names(l) <- c("Class", paste0(metric_names, "_train"), paste0(metric_names, "_test"))
correlations_p <- data.frame(Model = names(d[, -1]))
correlations_b <- data.frame(Model = names(d[, -1]))

outlier_removed = list()
remove_outliers = FALSE

spearmans_rho <- matrix(nrow = ncol(d) - 1, ncol = ncol(l) - 1)
colnames(spearmans_rho) <- names(l)[-1]
rownames(spearmans_rho) <- names(d)[-1]

spearman_p <- spearmans_rho

for(metric in names(l[, -1])){
  cor_p <- c()
  cor_b <- c()
  rcor_b <- c()
  for(model in names(d[, -1])){
    F1 <- d[, model]
    lmetr <- l[, metric]
    lin_cor <- lm(F1 ~ lmetr)
    p <- summary(lin_cor)$coefficients[2, 4]
    rlin_cor <- rlm(F1 ~ lmetr)
    spearman <- cor.test(lmetr, F1, method = "spearman")
    
    spearmans_rho[model, metric] <- spearman$estimate
    spearman_p[model, metric] <- spearman$p.value
    
    outliers <- which(hatvalues(lin_cor) > (mean(hatvalues(lin_cor)) + 2 * sd(hatvalues(lin_cor))))
    
    if(length(outliers) > 0 & remove_outliers){
        lin_cor <- lm(F1[-outliers] ~ lmetr[-outliers])
        outlier_removed[[length(outlier_removed) + 1]] <- paste(model, metric, "n = ", length(outliers))
    }
    
    # Test for heteroskedasticity using the Breusch Pagan test (homoscedasticity if p > 0.05)
    bpt <- ols_test_breusch_pagan(lin_cor)
    bpt.p <- round(summary(lin_cor)$coefficients[2, 4], 2)
    
    if(bpt.p < 0.05){
        print(paste0("Heteroscedasticity detected for ", model, " and ", metric, "."))
    }
    
    # Test normality of the residuals
    shapwilk <- shapiro.test(lin_cor$residuals)
    sw.p <- shapwilk$p.value
    
    if(sw.p < 0.05){
        print(paste0("Non-normality detected for ", model, " and ", metric, "."))
    }
    
    # Test for high-leverage values
    ols_plot_resid_stud(lin_cor) 
    ols_plot_resid_lev(lin_cor)
    
    # Plot to check for non-linear correlations
    plot(F1 ~ lmetr, main = paste0(model, ", ", metric, ", p = ", p))
    abline(a = lin_cor$coefficients[1], b = lin_cor$coefficients[2], col = "red")
    summary(lin_cor)
    cor_p <- c(cor_p, p)
    cor_b <- c(cor_b, lin_cor$coefficients[2])
    #rcor_b <- c(rcor_b, rlin_cor$coefficients[2])
  }
  correlations_p[, metric] <- cor_p
  correlations_b[, metric] <- cor_b
}

# Write appendix tables with full model details
sigfill <- function(x, sigfigs = 3){
  out <- gsub("\\.$", "",
              formatC(signif(x, digits = sigfigs),
                      digits = sigfigs, format = "fg", flag = "#"))
  out[grepl(".", out, fixed = TRUE)] <- strtrim(out[grepl(".", out, fixed = TRUE)],
                                                sigfigs + c(1, 2)[grepl("-", out, fixed = TRUE) + 1])
  return(out)
}

full_table <- data.frame(Model = correlations_p$Model)

for(j in 2:ncol(correlations_p)){
  full_table[, j] <- paste(
    sigfill(correlations_b[, j], 3), sigfill(correlations_p[, j], 3), sep = " & "
  )
}

full_table[, 1] <- c(rep(c(256, 512, 1024), 3), 512, 512)
full_table <- cbind(c("U-Net", " ", " ", "FC-DenseNet", " ", " ", "DeepLabv3+", " ", " ", "DL 2021", "DL 2022"),
                    full_table)

names(full_table) <- seq(1, ncol(full_table))

## Train ds
LSM_train <- tempfile()
write.table(full_table[, c(1:5)], file = LSM_train,
            col.names = c("\\multirow{2}{*}{{Model}}", "\\multirow{2}{*}{{Tile size}}",
                          "\\multicolumn{2}{c}{Mean patch area}",
                          "\\multicolumn{2}{c}{Class area}",
                          "\\multicolumn{2}{c}{Compactness}"),
            row.names = FALSE, sep = " & ", eol = "\\\\\n", quote = FALSE)
tmp0 <- readLines(LSM_train)
writeLines(paste(c("\\toprule",
                tmp[1],
                "& & {slope} & {\\textit{p}-value} & {slope} & {\\textit{p}-value} & {slope} & {\\textit{p}-value} \\\\",
                "\\midrule",
                tmp[-1],
                "\\bottomrule"), collapse = "\n"),
           con = LSM_train)
## Test ds
LSM_test <- tempfile()
write.table(full_table[, c(1, 2, 6:8)], file = LSM_test,
            col.names = c("\\multirow{2}{*}{{Model}}", "\\multirow{2}{*}{{Tile size}}",
                          "\\multicolumn{2}{c}{Mean patch area}",
                          "\\multicolumn{2}{c}{Class area}",
                          "\\multicolumn{2}{c}{Compactness}"),
            row.names = FALSE, sep = " & ", eol = "\\\\\n", quote = FALSE)
tmp1 <- readLines(LSM_test)
writeLines(paste(c("\\toprule",
                   tmp[1],
                   "& & {slope} & {\\textit{p}-value} & {slope} & {\\textit{p}-value} & {slope} & {\\textit{p}-value} \\\\",
                   "\\midrule",
                   tmp[-1],
                   "\\bottomrule"), collapse = "\n"),
           con = LSM_test)

## create combined table
writeLines(
  paste(c("\\begin{table}[hbtp]\n
     \\caption[Correlations between F1-Score and class properties]{Spearman's slope and \\textit{p}-value of linear models relating F1-Score and tree class properties as expressed through the metrics mean patch area (in ha), total class area (in ha) and compactness of patches (mean smallest circumscribing circle for patches of the class). Degrees of freedom = 14.}\n
     \\label{tab:LSM_full}\n
    \\begin{subtable}[h]{\\textwidth}\n
    \\caption{Training data}\n
       \\label{tab:LSMtraining}\n
        \\centering\n
        \\begin{tabular}{lrS[table-format=-3.2]S[table-format=1.2]S[table-format=1.2]S[table-format=1.2]S[table-format=-1.2]S[table-format=1.2]}",
          "\\toprule",
          tmp0[1],
          "& & {slope} & {\\textit{p}-value} & {slope} & {\\textit{p}-value} & {slope} & {\\textit{p}-value} \\\\",
          "\\midrule",
          tmp0[-1],
          "\\bottomrule",
          "\\end{tabular}\n
          \\end{subtable}\n
          \\begin{subtable}[h]{\\textwidth}\n
          \\caption{Test data}\n
          \\label{tab:LSMtest}\n
          \\centering\n
          \\begin{tabular}{lrS[table-format=3.2]S[table-format=1.2]S[table-format=-1.2]S[table-format=1.2]S[table-format=-1.2]S[table-format=1.2]}",
          "\\toprule",
          tmp1[1],
          "& & {slope} & {\\textit{p}-value} & {slope} & {\\textit{p}-value} & {slope} & {\\textit{p}-value} \\\\",
          "\\midrule",
          tmp1[-1],
          "\\bottomrule",
          "\\end{tabular}\n
     \\end{subtable}\n
\\end{table}"
        ), collapse = "\n"),
  con = file.path(dir_dropbox, "/tab/LSM_full.tex")
)

# Add asterisks for summary table in text body
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

#############################################
##### Load packages, set theme
#############################################
packages <- c("readxl", "ggplot2", "reshape2", "landscapemetrics", "dplyr")
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

#############################################
##### Load data
#############################################
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
}

plot(dat$D512[c(1:17)] ~ summary_lsm$mean_pa)
lin_mod <- lm(dat$D512[c(1:17)] ~ summary_lsm$mean_pa)
summary(lin_mod)

plot(dat$D512[c(1:17)] ~ summary_lsm$total_ca)
lin_mod <- lm(dat$D512[c(1:17)] ~ summary_lsm$total_ca)
summary(lin_mod)

plot(dat$D512[c(1:17)] ~ summary_lsm$mean_cm)
lin_mod <- lm(dat$D512[c(1:17)] ~ summary_lsm$mean_cm)
summary(lin_mod)

# insert some ANOVA stuff here

dl <- melt(dat, id.vars = "Species")
dl$value <- as.numeric(dl$value)
dl$Network <- rep(NA, nrow(dl))
dl$Tilesize <- rep(NA, nrow(dl))

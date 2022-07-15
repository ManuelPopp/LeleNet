#############################################
##### Load packages
#############################################
packages <- c("raster", "ggplot2", "vegan", "dplyr", "ggdendro", "stringr")
for(i in 1:NROW(packages)){
  if(!require(packages[i], character.only = TRUE)){
    install.packages(packages[i])
    library(packages[i], character.only = TRUE)
  }
}

#############################################
##### Set directories
#############################################
# Note: In order to get all class frequencies, DataPrep_grouped.bat
# should be ran with an .xlsx file explicitly assigning all species
# subsequent integer values, so all species are considered and no
# shifts occurr through automatical shifting of species values in
# case of gaps.
wd <- "G:/Masterarbeit/dat/omk/03_2021"

names <- c(paste0("ortho_B1_", seq(1, 6)),
           paste0("ortho_B2_", seq(1, 6)),
           paste0("ortho_B3_", seq(1, 6)),
           "ortho_B4_1", "ortho_B5_5", "ortho_B6_2", "ortho_B7_5", "ortho_B8_5")

#############################################
##### Set style options
#############################################
# graphics width mm for graphics device input (expects inch)
w_90mm <- 3.54331

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
  rgb(25, 161, 224, alpha = 254.99, maxColorValue = 255)# kit cyan
)
colz <- c(
  rgb(0, 150, 130, alpha = 255*0.5, maxColorValue = 255), #kit colour
  rgb(70, 100, 170, alpha = 255*0.5, maxColorValue = 255), #kit blue
  rgb(223, 155, 27, alpha = 255*0.5, maxColorValue = 255), #kit orange
  rgb(140, 182, 60, alpha = 255*0.5, maxColorValue = 255), #kit Mai green
  rgb(162, 34, 35, alpha = 255*0.5, maxColorValue = 255), #kit red
  rgb(163, 16, 124, alpha = 255*0.5, maxColorValue = 255), #kit violet
  rgb(167, 130, 46, alpha = 255*0.5, maxColorValue = 255) #kit brown
)

#############################################
##### Calculate class frequencies
#############################################
# Note: The following steps might take quite some time to calculate
frequencies <- list()

n_classes <- 67
for(i in 1:length(names)){
  path <- file.path(wd, paste0(names[i], "_MASK.tif"))
  mask <- raster::raster(path)
  values <- raster::freq(mask)
  total <- sum(values[, 2])
  values_perc <- rep(0, n_classes)
  values_perc[match(values[, 1], seq(0, n_classes))] <- values[, 2] / total
  frequencies[[names[i]]] <- values_perc
}

abundance_matrix <- matrix(unlist(frequencies), ncol = length(values_perc), byrow = TRUE)
row.names(abundance_matrix) <- names(frequencies)
colnames(abundance_matrix) <- seq(0, ncol(abundance_matrix) - 1)

#############################################
##### Class frequencies across all plots
#############################################
class_frequencies <- colSums(abundance_matrix) / nrow(abundance_matrix)
hist(class_frequencies, col = cols[1])

class_frequencies_df = data.frame(share = class_frequencies)
gg_hist <- ggplot(data = class_frequencies_df, aes(x = share)) +
  geom_histogram(fill = cols[2], binwidth = 0.02) +
  xlab("Share on total cover") +
  ylab("Number of classes")
gg_hist

w <- w_90mm
h <- w * 3 / 4

pdf("D:/Dateien/Studium_KIT/Master_GOEK/Masterarbeit/fig/ClassFreqHistA.pdf",
    width = w, height = h)
gg_hist
dev.off()

seq(0, n_classes)[which(class_frequencies > 0.01)]
class_frequencies_wout_soil <- class_frequencies[-length(class_frequencies)] /
  sum(class_frequencies[-length(class_frequencies)])

lower_boundary <- 0.01
main_classes <- seq(0, n_classes)[which(class_frequencies_wout_soil > lower_boundary)]
main_classes

#############################################
##### Plot similarities
#############################################
columns <- colnames(abundance_matrix) %in% as.character(main_classes)
abundance_matr <- abundance_matrix[, columns]
rownames(abundance_matr) <- str_replace(rownames(abundance_matr), "ortho_", "")

abm_norm <- decostand(abundance_matr, "normalize")
abm_ch <- vegdist(abm_norm, "euc")
attr(abm_ch, "labels") <- rownames(abundance_matr)

ab_single <- hclust(abm_ch, method = "single")
tree <- reorder(ab_single, wts = c(1, 2, 19, 20, 21, 22, 23, seq(3, 18)))
ggdendrogram(ab_single, rotate = FALSE, size = 2)
dendro <- ggdendrogram(tree)

pdf("D:/Dateien/Studium_KIT/Master_GOEK/Masterarbeit/fig/Dendrogram.pdf",
    width = w * 2, height = h)
dendro
dev.off()
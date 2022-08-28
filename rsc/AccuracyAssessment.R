#############################################
##### Load packages
#############################################
packages <- c("raster", "caret", "readxl")
for(i in 1:NROW(packages)){
  if(!require(packages[i], character.only = TRUE)){
    install.packages(packages[i])
    library(packages[i], character.only = TRUE)
  }
}

#############################################
##### Get rasters
#############################################
# select tilesize and block
tilesize <- 512
model = "U"

gt <- c()
pd <- c()
for(i in 1:3){
  plot <- c("B4_1", "B6_2", "B7_5")[i]
  msk_tif_name <- paste("ortho", plot, "MASK.tif", sep = "_")
  prd_tif_name <- paste("ortho", plot, "PRED.tif", sep = "_")
  path <- paste0("G:/Masterarbeit/dat/test_", model, tilesize)

  ground_truth <- raster::raster(file.path(path, msk_tif_name))
  prediction <- raster::raster(file.path(path, prd_tif_name))

  offset <- 128
  r <- nrow(ground_truth)
  c <- ncol(ground_truth)
  orig_ex <- raster::extent(ground_truth)
  crop_ex <- raster::extent(ground_truth,
                            offset, (r - offset),
                            offset, (c - offset))
  plot(prediction)
  plot(orig_ex, add = TRUE, col = "red")
  plot(crop_ex, add = TRUE, col = "blue")
  
  ground_truth <- raster::crop(ground_truth, crop_ex)
  prediction <- raster::crop(prediction, crop_ex)
  
  ground_truth <- raster::resample(ground_truth, prediction, method = "ngb")
  
  gt <- append(gt, as.vector(ground_truth))
  pd <- append(pd, as.vector(prediction))
}

#############################################
##### Calculate confusion matrix
#############################################
lvls <- sort(union(gt, pd))
CM <- caret::confusionMatrix(data = factor(pd, lvls),
                             reference = factor(gt, lvls))
confmatr <- CM$table
accuracy <- CM$overall["Accuracy"]
kappa <- CM$overall["Kappa"]
precision <- CM$byClass[, "Precision"]
recall <- CM$byClass[, "Recall"]
F1Score <- CM$byClass[, "F1"]

#############################################
##### Create table
#############################################
# get classnames
SpeciesList_path <- "G:/Masterarbeit/dat/xls/SpeciesList.xlsx"
SpeciesList <- readxl::read_excel(SpeciesList_path, sheet = "Dictionary")
classes <- sort(unique(SpeciesList$Class))
classnames <- data.frame(Class = classes,
                         Name = rep(NA, length(classes)),
                         Level = rep(NA, length(classes)))
for(i in 1:nrow(classnames)){
  class <- classnames$Class[i]
  N <- length(which(SpeciesList$Class == class))
  if(N == 1){
    classnames$Name[i] <- SpeciesList$Species_Name[which(SpeciesList$Class == class)]
    classnames$Level[i] <- "Species"
  }else{
    genera <- unique(SpeciesList$Genus[which(SpeciesList$Class == class)])
    if(length(genera) == 1){
      classnames$Name[i] <- paste(genera, "spec.")
      classnames$Level[i] <- "Genus"
    }else if(length(genera) > 1){
      classnames$Name[i] <- "Other"
      classnames$Level[i] <- "Kingdom"
    }
  }
}
classnames[nrow(classnames) + 1,] <- c((max(classnames$Class) + 1), NA, NA)
classnames[nrow(classnames), c(2, 3)] <- c("Background", "None")

# create table
acc_table <- data.frame(Class = classnames$Name, ResultF1 = F1Score)
# write to clipboard (paste to Excel)
writeClipboard(acc_table$Class)
writeClipboard(as.character(precision))
writeClipboard(as.character(recall))
writeClipboard(as.character(acc_table$ResultF1))
kappa
accuracy
colnames(confmatr) <- classnames$Name
row.names(confmatr) <- classnames$Name
write.table(confmatr, paste0("G:/Masterarbeit/dat/xls/", model, tilesize, ".txt"),
            sep = "\t", row.names = TRUE)

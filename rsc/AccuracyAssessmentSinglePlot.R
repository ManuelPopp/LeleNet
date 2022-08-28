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
path <- "G:/Masterarbeit/dat/test_D1024"
msk_tif_name <- "ortho_B7_5_MASK.tif"
prd_tif_name <- "ortho_B7_5_PRED.tif"

ground_truth <- raster::raster(file.path(path, msk_tif_name))
prediction <- raster::raster(file.path(path, prd_tif_name))

tilesize <- 1024
offset <- tilesize/1.5
r <- nrow(prediction)
c <- ncol(prediction)
orig_ex <- raster::extent(prediction)
crop_ex <- raster::extent(prediction,
                          offset, (r - offset),
                          offset, (c - offset))

plot(prediction)
plot(orig_ex, add = TRUE, col = "red")
plot(crop_ex, add = TRUE, col = "blue")

ground_truth <- raster::crop(ground_truth, crop_ex)
prediction <- raster::crop(prediction, crop_ex)

#############################################
##### Calculate confusion matrix
#############################################
gt <- as.vector(ground_truth)
pd <- as.vector(prediction)
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
writeClipboard(as.character(acc_table$ResultF1))
accuracy

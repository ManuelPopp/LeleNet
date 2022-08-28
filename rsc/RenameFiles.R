directory <- "F:/"
block <- "1"
plot <- "6"
folder <- paste(directory, "Block_", block, "/Block", block, "_", plot, sep = "")
imgs <- list.files(folder, full.names = TRUE, pattern = ".JPG")
#for(i in 1:length(imgs)){
#  date <- file.info(imgs[[i]])$ctime
#  file.rename(imgs[[i]], paste("/home/manuel/Desktop/Block08/100MEDIA/", "Img", i, date, ".jpg", sep = ""))
#}
substR <- function(x, n){
  substr(x, nchar(x) - n + 1, nchar(x))
}

# give each image a unique name recognisable across all folders
for(img in imgs){
  name <- substR(img, 8)
  file.rename(img, paste(folder, "/B", block, "_", plot, "_", name, sep = ""))
}

path <- "C:/Users/Manuel/Nextcloud/Masterarbeit/dat/tls/Test/X0"
outpath <- "C:/Users/Manuel/Nextcloud/Masterarbeit/dat/tls/Test/X/1"
files <- list.files(path, pattern = ".tif", full.names = TRUE)
names <- list.files(path, pattern = ".tif", full.names = FALSE)
require("raster")
for(i in 1:length(files)){
  rast <- stack(files[i])
  writeRaster(rast, filename = paste(outpath, names[i], sep = "/"), overwrite = TRUE, format = "GTiff", datatype = "INT1U")
}

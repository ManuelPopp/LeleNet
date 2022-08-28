#############################################
##### Set directories
#############################################
if(Sys.info()['sysname'] == "Windows"){
  # paths Win
  wd <- "C:/Users/Manuel/Nextcloud/Masterarbeit/"
}else if(Sys.info()['sysname'] == "Linux"){
  # paths Lin
  wd <- "/home/manuel/Nextcloud/Masterarbeit/"
}else{
  print("Error: OS not identified.")
}

shp.dir <- paste(wd, "/gis/QGIS/Shapefiles/", sep = "")

#############################################
##### Load packages
#############################################
packages <- c("raster", "sp", "sf", "leafem", "leafpop", "mapview", "rgdal", "rgeos")#
for(i in 1:NROW(packages)){
  if (!require(packages[i], character.only= TRUE)) {
    install.packages(packages[i])
    library(packages[i], character.only= TRUE)
  }
}

#############################################
##### Load Block shapefiles
#############################################
blocks <- readOGR(dsn = paste(shp.dir, "Blocks.shp", sep = ""))
proj4string(blocks)
blocks <- spTransform(
  blocks, CRS("+proj=utm +zone=35 +south +datum=WGS84 +units=m +no_defs")
  )

#############################################
##### Dissolve and buffer Block shapefiles
#############################################
subplots <- list()
for(i in 1:length(blocks)){
  subplots[[i]] <- blocks[i,]
  names(subplots)[i] <- blocks@data$Name[i]
}

buffer <- 11
subplots.buff <- list()
for(i in 1:length(blocks)){
  #subplots.buff[[i]] <- gBuffer(subplots[[i]], width = buffer)
  adjacent <- gCentroid(
    blocks[gTouches(
      blocks, byid = TRUE
      )[i,],], byid = TRUE
    )
  xbuff <- c(-buffer, -buffer, buffer, buffer, buffer)
  ybuff <- c(-buffer, buffer, buffer, -buffer, -buffer)
  xcoords <- extent(subplots[[i]])[c(1, 1, 2, 2, 1)] + xbuff
  ycoords <- extent(subplots[[i]])[c(3, 4, 4, 3, 3)] + ybuff
  coords <- cbind(xcoords, ycoords)
  tmp <- Polygon(coords)
  tmp <- Polygons(list(tmp), 1)
  subplots.buff[[i]] <- SpatialPolygons(
    list(tmp), pO = integer(1),
    CRS("+proj=utm +zone=35 +south +datum=WGS84 +units=m +no_defs")
    )
}
names(subplots.buff) <- names(subplots)

#############################################
##### Export subplots as kml
#############################################
flight.dir <- paste(wd, "dji/FlightPlans/", sep = "")
dir.create(flight.dir)

for(i in 1:length(blocks)){
  tmp <- SpatialPolygonsDataFrame(
    spTransform(
      subplots.buff[[i]],
      CRS("+proj=longlat +datum=WGS84 +no_defs")
      ),
    data.frame(x = names(subplots.buff)[i], row.names = slot(subplots.buff[[i]]@polygons[[1]], "ID"))
    )
  writeOGR(tmp, dsn = paste(flight.dir, names(subplots.buff)[i], ".kml", sep = ""),
           layer = 1, driver = "KML", overwrite_layer = TRUE)
}
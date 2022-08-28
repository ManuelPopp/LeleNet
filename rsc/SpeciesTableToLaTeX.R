#############################################
##### Set directories
#############################################
if(Sys.info()['sysname'] == "Windows"){
  # paths Win
  wd <- "C:/Users/Manuel/Nextcloud/Masterarbeit/"
  db <- "C:/Users/Manuel/Dropbox/Apps/Overleaf/Masterarbeit"
}else if(Sys.info()['sysname'] == "Linux"){
  # paths Lin
  wd <- "/home/manuel/Nextcloud/Masterarbeit/"
  db <- "/home/manuel/Dropbox/Apps/Overleaf/Masterarbeit"
}else{
  print("Error: OS not identified.")
}

dir_fig <- file.path(wd, "fig")

#############################################
##### Load packages
#############################################
packages <- c("sp", "sf", "dplyr", "WorldFlora", "xtable", "stringi")
for(i in 1:NROW(packages)){
  if(!require(packages[i], character.only = TRUE)){
    install.packages(packages[i])
    library(packages[i], character.only = TRUE)
  }
}

# ArcGIS connection
if(!require("arcgisbinding", character.only = TRUE)){
  install.packages("arcgisbinding", repos = "https://r.esri.com", type = "win.binary")
  library("arcgisbinding", character.only = TRUE)
}

#############################################
##### Load data from ArcGIS Online
#############################################
# ArcGIS Online data
arc.check_product()

Blocks <- arc.data2sf(
  arc.select(
    arc.open(
      "https://services9.arcgis.com/ogIE9wkhkJFrskxj/arcgis/rest/services/LELEBlockOutline/FeatureServer/0"
    )
  )
)[, c("OBJECTID", "LELE_Block", "Block", "Plot", "Treatment")]
#Blocks <- st_transform(Blocks, crs = "+proj=longlat +datum=WGS84")

Trees <- arc.data2sf(
  arc.select(
    arc.open(
      "https://services9.arcgis.com/ogIE9wkhkJFrskxj/arcgis/rest/services/Trees_from_scratch/FeatureServer/0"
    )
  )
)[, c("OBJECTID", "Species", "TreeDiameter1", "SpeciesID")]
Trees <- st_transform(Trees, crs = st_crs(Blocks))
Trees <- Trees[Trees$Species != "Other",]

# Shapefiles
require("rgdal")
Lapalala <- readOGR(file.path(wd, "gis", "QGIS", "Shapefiles", "Lapalala", "LapalalaBorders.shp"))

TreesWithPlots <- st_join(x = Trees, y = Blocks)

#############################################
##### Print species list
#############################################
options(timeout = max(300000, getOption("timeout")))
WFO.download(WFO.url = "http://104.198.143.165/files/WFO_Backbone/_WFOCompleteBackbone/WFO_Backbone.zip",
             save.dir = tempdir(), WFO.remember = TRUE)
WFO.remember(WFO.file = file.path(tempdir(), "WFO_Backbone.zip"), WFO.data = "WFO.data", WFO.pos = 1)
dir.create(file.path(tempdir(), "WFO_Backbone"), showWarnings = FALSE)
setwd(file.path(tempdir(), "WFO_Backbone"))
unzipped <- unzip(file.path(tempdir(), "WFO_Backbone.zip"))
WFO.file.RK <- file.path(tempdir(), "WFO_Backbone", "classification.txt")
WFO.data1 <- data.table::fread(WFO.file.RK, encoding = "UTF-8")

all_species <- unique(TreesWithPlots$Species)
all_species[which(all_species == "Euclea linearis")] <- "Euclea crispa"

df_all_spec <- data.frame(Family = rep(NA, length(all_species)),
                          Genus = rep(NA, length(all_species)),
                          Epithet = rep(NA, length(all_species)),
                          Authority = rep(NA, length(all_species)),
                          Source = rep(NA, length(all_species)))

get_infos <- function(Species = NULL){
  WFO.browse(Species, WFO.data = WFO.data1, accepted.only = TRUE)
  spec_info <- WFO.match(Species, WFO.data = WFO.data1)
  return(c(spec_info$family[1], spec_info$genus[1], spec_info$specificEpithet[1],
           spec_info$scientificNameAuthorship[1],
           spec_info$namePublishedIn[1]))
}

for(i in 1:length(all_species)){
  df_all_spec[i, ] <- get_infos(all_species[i])
}

df_all_spec$Genus <- paste0("openitshape", df_all_spec$Genus, "closeitshape")
df_all_spec$Epithet <- paste0("openitshape", df_all_spec$Epithet, "closeitshape")

df_all_spec[which(df_all_spec$Epithet == "openitshapecrispacloseitshape")[1],
            c("Epithet", "Authority", "Source")] <- c(
              "openitshapecrispacloseitshape subsp. openitshapelineariscloseitshape",
              "(Zeyh. ex Hiern) F.White",
              "Bull. Jard. Bot. Natl. Belg.")

# order by species name
df_all_spec <- df_all_spec[with(df_all_spec,
                                order(df_all_spec$Genus, df_all_spec$Epithet)), ]

print(xtable(df_all_spec,
             caption = "List of all tree species found in the research plots. Taxon names and information were semiautomatically matched and manually checked following World Flora Online (Kindt, 2021; WFO, 2022).",
             label = "tab:Secies_list"),
      table.placement = "pbht",
      caption.placement = "top",
      include.rownames = FALSE,
      booktabs = TRUE,
      file = file.path(db, "tab", "Species_List.tex"),
      add.to.row = list(list(nrow(df_all_spec)),  
                        '\\bottomrule\n\\multicolumn{5}{l}{Kindt, Roeland (2020). \\enquote{WorldFlora: An R Package for Exact and Fuzzy Matching of Plant Names against
                        the World Flora Online Taxonomic Backbone Data}.}\\\\\n \\multicolumn{5}{l}{\\hspace{1em} In: openitshapeApplications in Plant Sciencescloseitshape 8. 9, e11388. issn: 2168-0450. doi: \\href{https://doi.org/10.1002/aps3.11388}{10.1002/aps3.11388}.}\\\\\n \\multicolumn{5}{l}{WFO (2022): World Flora Online. Published on the Internet; \\url{http://www.worldfloraonline.org}. Accessed on: 06 Feb 2022.}\\\\'))
# adjust table size
text_file <- readLines(file.path(db, "tab", "Species_List.tex"))
text_file <- c(text_file[1:6], "\\resizebox{\\textwidth}{!}{", text_file[7:length(text_file)])
text_file <- c(text_file[1:length(text_file)-1], "}", text_file[length(text_file)])
text_file <- gsub("1947 \\[Sep 1947\\]", "(1947)", text_file, perl = TRUE)
text_file <- gsub("caption", "caption[Species list]", text_file)
text_file <- gsub("ä", '\\\\"a', text_file)
text_file <- gsub("ü", '\\\\"u', text_file)
text_file <- gsub("é", "\\\\'e", text_file)
text_file <- stri_replace_last(text_file, fixed = "\\\\ \\bottomrule", replacement = "\\\\")
text_file <- gsub("openitshape", "\\\\textit\\{", text_file)
text_file <- gsub("closeitshape", "\\}", text_file)
writeLines(text_file, file.path(db, "tab", "Species_List.tex"))
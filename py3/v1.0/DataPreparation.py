#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 17 17:56:34 2021

@author: Manuel
Set general functions and variables.
"""
__author__ = "Manuel R. Popp"

trn_plots = []
'''
["B1_6_0003", "B1_6_0023", "B1_6_0063", "B1_6_0078", "B1_6_0086",\
             "B1_6_0119", "B1_6_0136", "B1_6_0140", "B1_6_0151", "B1_6_0181",\
             "B1_6_0198", "B1_6_0209", "B1_6_0239", "B1_6_0256", "B1_6_0428",\
             "B1_6_0447", "B1_6_0480", "B1_6_0417", "B1_6_0498", "B2_2_0043",\
             "B2_2_0053", "B2_2_0096", "B2_2_0123", "B2_2_0157", "B2_2_0166",\
             "B3_4_0084", "B3_4_0096", "B3_4_0213", "B3_4_0230", "B4_5_0053",\
             "B4_5_0061", "B4_5_0157", "B4_5_0166", "B5_4_0550", "B5_4_0560",\
             "B5_4_0657", "B5_4_0665", "B6_4_0554", "B6_4_0560", "B6_4_0624",\
             "B6_4_0650", "B7_6_0452", "B7_6_0463", "B8_4_0080", "B8_4_0088"]#, "B7_6_0583", "B7_6_0590", "B8_4_0192", "B8_4_0205"]
'''
val_plots = []
'''
["B1_6_0286", "B1_6_0267", "B2_2_0061", "B2_2_0131", "B2_2_0148",\
             "B3_4_0065", "B3_4_0192", "B4_5_0044", "B4_5_0149", "B5_4_0542",\
             "B5_4_0647", "B6_4_0544", "B6_4_0615",  "B7_6_0482", "B8_4_0060"]#, "B7_6_0598", "B8_4_0174"]
'''
tst_plots = ["ortho_B1_2", "ortho_B1_3", "ortho_B1_4", "ortho_B1_5"]
all_plots = trn_plots + val_plots + tst_plots

#### parse arguments-----------------------------------------------------------
import argparse
import urllib.request # in case it isn't loaded properly by arcgis

def parseArguments():
    parser = argparse.ArgumentParser()
    # Optional arguments
    parser.add_argument("-date", "--date",\
                        help = ("Sampling date of the data" +\
                                "as MM_YYYY. Default: '03_2021'"),\
                            type = str, default = "03_2021")
    parser.add_argument("-xf", "--xf",\
                        help = "RGB image format; either png, jpg, or tif.",\
                            type = str, default = "png")
    parser.add_argument("-yf", "--yf",\
                        help = "Mask image format; either png, jpg, or tif.",\
                            type = str, default = "png")
    parser.add_argument("-imgr", "--imgr",\
                        help = "Image x resolution (rows).", type = int,\
                            default = 256)
    parser.add_argument("-imgc", "--imgc",\
                        help = "Image y resolution (columns).", type = int,\
                            default = 256)
    parser.add_argument("-imgd", "--imgd",\
                        help = "Image y resolution (square).", type = int,\
                            default = None)
    parser.add_argument("-ndc", "--ndc",\
                        help = "No data class.", type = bool,\
                            default = False)
    parser.add_argument("-abc", "--abc",\
                        help = "Additional background class.", type = int,\
                            default = -1)
    parser.add_argument("-grp", "--grp",\
                        help = "Group species by dictionary.", type = bool,\
                            default = False)
    parser.add_argument("-mbd", "--mbd",\
                        help = "Mark bad data in additional mask image" +\
                            "dimension.", type = bool, default = False)
    parser.add_argument("-wd", "--wd",\
                        help = "Alternative working directory.", type = str,\
                            default = "")
    parser.add_argument("-add", "--add",\
                        help = "Add data for new plots without deleting ol" +\
                        "d data.", type = bool,\
                            default = True)
    # Parse arguments
    args = parser.parse_args()
    return args
if __name__ == "__main__":
    # Parse the arguments
    args = parseArguments()

xf = args.xf
yf = args.yf
xf, yx = xf.casefold(), yf.casefold()
imgr = args.imgr
imgc = args.imgc
imgd = args.imgd
date = args.date
no_data_class = args.ndc
additional_background_class = args.abc
abc = True if additional_background_class >= 0 else False
group_species = args.grp
mark_bad_data = args.mbd
wd = "home"
wd = args.wd
just_add = args.add
extent_from_shapefile = False

if imgd is not None:
    imgr = imgd
    imgc = imgd

#### basic settings------------------------------------------------------------
import platform, sys, datetime
OS = platform.system()
OS_version = platform.release()
py_version = sys.version
t_start = datetime.datetime.utcnow()
print("Running on " + OS + " " + OS_version + ".\nPython version: " +
      py_version +
      "\nUTC time (start): " + str(t_start) +
      "\nLocal time (start): " + str(datetime.datetime.now()))

### general directory functions------------------------------------------------
import os, pickle
import numpy as np
if wd == "home":
    if OS == "Linux":
        wd = "/home/manuel/Nextcloud/Masterarbeit"
    elif OS == "Windows":
        wd = os.path.join("G:\\", "Masterarbeit")
    else:
        raise Exception("OS not detected.")
elif wd == "":
    pydir = os.path.dirname(os.path.realpath(__file__))
    wd = os.path.dirname(pydir)
else:
    wd = args.wd

print("wd: " + wd)

def dir_fig(fig_id = None):
    if fig_id == None:
        return os.path.join(wd, "fig")
    else:
        return os.path.join(wd, "fig", fig_id)

def dir_dat(dat_id = None):
    if dat_id == None:
        return os.path.join(wd, "dat")
    else:
        dat_id = dat_id.split(",")
        return os.path.join(wd, "dat", *dat_id)

def dir_out(out_id = None):
    if out_id == None:
        return os.path.join(wd, "out")
    else:
        return os.path.join(wd, "out", out_id)

def dir_var(pkl_name = None):
    if pkl_name == None:
        return os.path.join(wd, "py3", "vrs")
    else:
        return os.path.join(wd, "py3", "vrs", pkl_name + ".pkl")

def save_var(variables, name):
    os.makedirs(dir_var(), exist_ok = True)
    with open(dir_var(pkl_name = name), "wb") as f:
        pickle.dump(variables, f)

def get_var(name):
    with open(dir_var(pkl_name = name), "rb") as f:
        return pickle.load(f)
    
os.chdir(wd)


#### data preparation directory functions--------------------------------------
import pathlib
def dir_omk(plot_id = None, mdate = None, type_ext = ""):
    # returns list!
    if plot_id == None:
        if mdate == None:
            return dir_dat("omk")
        else:
            return os.path.join(dir_dat("omk"), mdate)
    else:
        if mdate == None:
            return list(pathlib.Path(dir_dat("omk")) \
                        .glob("**/*" + plot_id + type_ext + ".tif"))
        else:
            return list(pathlib.Path(os.path.join(dir_dat("omk"), mdate)) \
                        .glob("**/*" + plot_id + type_ext + ".tif"))

def dir_tls(mdate = None, dset = None, plot_id = None):
    if plot_id == None:
        if mdate == None:
            if dset == None:
                return dir_dat("tls")
            else:
                return dir_dat("tls")
                raise Exception("Missing date. Returning tile directory.")
        else:
            if dset == None:
                return os.path.join(dir_dat("tls"), mdate)
            else:
                return os.path.join(dir_dat("tls"), mdate, dset, "0")
    else:
        if mdate == None:
            return dir_dat("tls")
            raise Exception("Missing date. Returning tile directory.")
        else:
            if dset == None:
                return os.path.join(dir_dat("tls"), mdate)
                raise Exception("Missing dset (X or y). " + \
                                "Returning tile directory.")
            else:
                return os.path.join(dir_dat("tls"), mdate, dset, "0", plot_id)

def save_dataset_info(variables, date = date, name = "dset_info"):
    tile_dir = dir_tls(mdate = date)
    os.makedirs(tile_dir, exist_ok = True)
    with open(tile_dir + os.path.sep + name + ".pkl", "wb") as f:
        pickle.dump(variables, f)

def get_dataset_info(date = date, name = "dset_info"):
    tile_dir = dir_tls(mdate = date)
    with open(tile_dir + os.path.sep + name + ".pkl", "rb") as f:
        return pickle.load(f)

def toINT(filename):
    imgINT = filename.astype("uint8")
    return imgINT

# Data preparation-------------------------------------------------------------
###  read dictionary to group species to classes, if need be
import pandas as pd
specdict = pd.read_excel(dir_dat("xls,SpeciesList.xlsx"),
                         sheet_name = "Dictionary", header = 0)

# Define functions-------------------------------------------------------------
## AcrGIS Online
import arcgis, math
from arcgis.gis import GIS
gis = GIS("pro")
def dir_shp(plot_id = None, mdate = None):
    if mdate == None:
        if plot_id == None:
            return(dir_dat("shp"))
        else:
            print("Option 'mdate' missing but required.")
    else:
        if plot_id == None:
            return(os.path.join(dir_dat("shp"), mdate))
        else:
            os.makedirs(os.path.join(dir_dat("shp"), mdate, plot_id),
                        exist_ok = True)
            return(os.path.join(dir_dat("shp"), mdate, plot_id))

def downloadShapefiles(plot_id, path, dateTime = None):
    try:
        ## Search items by username
        cont = gis.content.search("owner:{0}".format("manuel.popp_KIT"),
                                  max_items = 3*len(trn_plots) + 999)
        for i in range(len(cont)):
            shp = cont[i]
            if plot_id in shp.title and shp.type == "Feature Service":
                if len(list(pathlib.Path(path). \
                             glob("**/*" + plot_id + ".shp"))) < 1:
                    update = True
                    print("Shapefile", plot_id, "not found.", \
                          "Downloading Feature Service from ArcGIS Online...")
                elif not datetime == None:
                    item = gis.content.get(shp.id)
                    data_modfd = max(
                        [j.properties.editingInfo.lastEditDate \
                        for j \
                        in item.layers + item.tables]\
                        )
                    update = data_modfd/1000 > dateTime
                    if update:
                        print("Updating Feature layer", plot_id + "...")
                    else:
                        print("Feature layer", plot_id, \
                              "already up-to-date.")
                    #somehow, ArcGIS timestamp has to be divided by 1000
                    #datetime.datetime.fromtimestamp(data_modfd/1000)
                    #datetime.datetime.fromtimestamp(dateTime)
                elif dateTime == "any":
                    update = True
                else:
                    update = False
                if update:
                    shapefile = shp.export("layer {}".format(shp.type),
                                           "Shapefile")
                    shapefile.download(path)
                    shapefile.delete()
                    import zipfile as zf
                    tmp = zf.ZipFile(
                        list(pathlib.Path(path). \
                             glob("**/*.zip"))[0],
                            mode = "r")
                    tmp.extractall(path = path)
                    tmp.close()
                    os.remove(list(pathlib.Path(path). \
                                   glob("**/*.zip"))[0])
    except Exception as e:
        print(e)

def get_classes(path, plots):
    from osgeo import ogr
    drv = ogr.GetDriverByName("ESRI Shapefile")
    ## read existing classes
    files = list(pathlib.Path(path).glob("**/*.shp"))
    folder = list(os.path.basename(file.parent) for file in files)
    plotfiles = list(np.where(np.isin(folder, plots)))
    files = [val for val, use in zip(files, np.isin(folder, plots)) if use]
    shapefiles = []
    for file in files:
        par = file.parents[0]
        plot = str(par.name)
        if str(plot + ".shp") in str(file):
            shapefiles.append(file)
    class_list = []
    for path_shp in shapefiles:
        dataSource = drv.Open(os.path.join(path_shp), 0)
        layer = dataSource.GetLayer()
        feature = layer.GetNextFeature()
        while feature:
            class_list.append(feature.GetField("SpeciesID"))
            feature = layer.GetNextFeature()
        feature = None
        layer = None
        dataSource = None
    class_set = set(class_list)
    classes = list(class_set)
    return(classes)

def encode_classes(path, classes, custom_dict = None):
    from osgeo import ogr
    drv = ogr.GetDriverByName("ESRI Shapefile")
    ## create new field
    dataSource = drv.Open(path, 1)
    FDef = ogr.FieldDefn("Class", ogr.OFTInteger)
    layer = dataSource.GetLayer()
    exists = layer.GetLayerDefn().GetFieldIndex("Class") >= 0
    if not exists:
        layer.CreateField(FDef)
    feature = layer.GetNextFeature()
    while feature:
        spec = feature.GetField("SpeciesID")
        c = classes.index(spec) if custom_dict is None else\
            custom_dict.loc[custom_dict.SpeciesID == spec, "Class"].item()
        feature.SetField("Class", c)
        layer.SetFeature(feature)
        feature = layer.GetNextFeature()
    dataSource.SyncToDisk()
    feature = None
    layer = None
    dataSource = None
    outdir = os.path.dirname(os.path.realpath(path))
    output = os.path.join(outdir, "class_encoding.csv")
    import csv
    with open(output, "w", newline = "") as out:
        wr = csv.writer(out)
        wr.writerows(zip(["ClassID"], ["Encoded"]))
        wr.writerows(zip(classes, range(len(classes))))

## get data and create masks and tiles
import numpy as np
from osgeo import gdal, ogr, osr
from osgeo.gdal_array import CopyDatasetInfo, BandReadAsArray
def blacken(data, mask, NoDataVal, outFile = "/vsimem/Blackened.tif"):
    dat = gdal.Open(data, gdal.GA_ReadOnly)
    msk = gdal.Open(mask, gdal.GA_ReadOnly)
    bm = msk.GetRasterBand(1)
    am = BandReadAsArray(bm)
    null = np.where(am == NoDataVal)
    am = None
    bm = None
    msk = None
    drv = gdal.GetDriverByName("GTiff")
    new = drv.Create(outFile, dat.RasterXSize, dat.RasterYSize,
                     dat.RasterCount, dat.GetRasterBand(1).DataType)
    CopyDatasetInfo(dat, new)
    for b in range(dat.RasterCount):
        band = dat.GetRasterBand(b+1)
        a = BandReadAsArray(band)
        a[null] = 0
        new.GetRasterBand(b+1).WriteArray(a)
        #new.GetRasterBand(b).SetNoDataValue(NoDataVal)
        a = None
    new.FlushCache()
    dat = None
    return new

def check_version(file, derived_from):
    ### expects pathlib path list
    if len(file) < 1:
        return None
    elif len(derived_from) < 1:
        print("Please pass a valid path to the derived_from argument.")
    else:
        file_mod = file[0].stat().st_mtime
        from_mod = derived_from[0].stat().st_mtime
        return (file_mod > from_mod)

# Update extents if required---------------------------------------------------
if not extent_from_shapefile:
    if not os.path.isfile(os.path.join(dir_shp("Extents", mdate = date),
                                   "Extents.shp")):
        last_modified = None
    else:
        last_modified = pathlib.Path(dir_shp("Extents", mdate = date),
                                   "Extents.shp").stat().st_mtime
    downloadShapefiles(plot_id = "Extents",
                               path = dir_shp("Extents", mdate = date),
                               dateTime = last_modified)
    Ext = os.path.join(dir_shp("Extents", mdate = date), "Extents.shp")
    ExtUTM = Ext[:len(Ext)-4] + "_UTM.shp"
    ## transform extents shapefile to UTM Zone 35S
    drv = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(ExtUTM):
        drv.DeleteDataSource(ExtUTM)
    srcDS = gdal.OpenEx(Ext)
    crs = osr.SpatialReference()
    crs.ImportFromEPSG(32735)
    ds = gdal.VectorTranslate(ExtUTM, srcDS, format = "ESRI Shapefile",
                              dstSRS = crs, reproject = True)
    ds = None
    srcDS = None
    ### write .prj file
    import re
    with open(f"{os.path.splitext(ExtUTM)[0]}.prj", "w") as f:
        f.write(re.sub(" +", " ",str(crs).replace("\n", "")))

# Iterate through image lists--------------------------------------------------
from PIL import Image#for comparison of image dimensions
for set_X, set_y, use_plots in zip(["X", "X_val", "X_tst"], \
                                   ["y", "y_val", "y_tst"], \
                                   [trn_plots, val_plots, tst_plots]):
    delete_old_tiles = True

    # run for each image or plot
    for plot in use_plots:
        ### update shapefile if required
        if len(list(pathlib.Path(dir_shp(plot, mdate = date)) \
                    .glob("**/*" + plot + ".shp"))) < 1:
            last_modified = None
        else:
            last_modified = list(pathlib.Path(dir_shp(plot, mdate = date)). \
                                 glob("**/*" + plot + \
                                      ".shp"))[0].stat().st_mtime
        downloadShapefiles(plot_id = plot,
                               path = dir_shp(plot, mdate = date),
                               dateTime = last_modified)
    ## get class IDs
    if set_X == "X":
        classes_decoded = get_classes(dir_shp(mdate = date), plots = all_plots)
        # NoDataValue = Class "bare soil"
        NoDataValue = len(classes_decoded) if not group_species else\
            int(specdict["Class"].max() + 1)
        classes = range(len(classes_decoded)) if not group_species else\
            range(int(specdict["Class"].max() + 1))
        if additional_background_class >= 0:
            NoDataValue = additional_background_class
            classes = range(len(classes_decoded)) if not group_species else\
                range(int(specdict["Class"].max() + 1))
        save_dataset_info(variables = [classes, classes_decoded, NoDataValue,\
                              no_data_class, abc])
    print("Using background value " + str(NoDataValue))
    
    for plot in use_plots:
        shp_path = list(pathlib.Path(dir_shp(plot, mdate = date)) \
                        .glob("**/*" + plot + ".shp"))[0]
        if group_species:
            encode_classes(path = os.path.join(shp_path),
                           classes = classes_decoded,
                           custom_dict = specdict)
        else:
            encode_classes(path = os.path.join(shp_path),
                           classes = classes_decoded)
        ### tif and shp path
        t = os.path.join(dir_omk(plot)[0])
        plib = pathlib.Path(dir_shp(plot, mdate = date))
        ### check if shapefile was transformed to UTM
        if check_version(file = list(pathlib.Path(dir_shp(plot, \
                                                          mdate = date)).\
                                  glob("**/*_UTM.shp")),
                      derived_from = list(
                          pathlib.Path(dir_shp(plot, mdate = date)).\
                                  glob("**/*" + plot + ".shp"))) in (None, \
                                                                     False):
            if os.path.exists(t):
                ### open orthomosaic
                tif = gdal.Open(t)
                crs1 = tif.GetProjectionRef()#GetSpatialRef()#.GetProjection()
                tif = None
                ### reproject shapefile
                s = os.path.join(list(plib.glob("**/*.shp"))[0])
                sUTM = s[:len(s)-4] + "_UTM.shp"
                drv = ogr.GetDriverByName("ESRI Shapefile")
                if os.path.exists(sUTM):
                    drv.DeleteDataSource(sUTM)
                srcDS = gdal.OpenEx(s)
                ds = gdal.VectorTranslate(sUTM, srcDS, format="ESRI Shapefile",
                                          dstSRS = crs1, reproject = True)
                ds = None
                srcDS = None
                ### write .prj file
                with open(f"{os.path.splitext(sUTM)[0]}.prj", "w") as f:
                    f.write(crs1)
            else:
                raise Exception("Raster is missing (for layout reference): "+t)
        if check_version(file = dir_omk(plot, type_ext = "_MASK"),
                         derived_from = list(pathlib.Path(dir_shp(plot,
                                                                  mdate=date)).\
                                  glob("**/*_UTM.shp"))) in (None, False):
            ## create mask
            sUTM = os.path.join(list(plib.glob("**/*_UTM.shp"))[0])
            drv = ogr.GetDriverByName("ESRI Shapefile")
            shp = drv.Open(sUTM)
            layer = shp.GetLayer()

            if extent_from_shapefile:
                x_min, x_max, y_min, y_max = (sys.maxsize, 0, sys.maxsize, 0)
                feature = layer.GetNextFeature()
                while feature:
                    geom = feature.GetGeometryRef()
                    extent = geom.GetEnvelope()
                    x_min = min(x_min, extent[0])
                    x_max = max(x_max, extent[1])
                    y_min = min(y_min, extent[2])
                    y_max = max(y_max, extent[3])
                    feature = layer.GetNextFeature()
                feature = None
                extent = (x_min, x_max, y_min, y_max)
            else:
                Ext = drv.Open(ExtUTM)
                lr = Ext.GetLayer()
                ft = lr.GetNextFeature()
                while ft:
                    if ft.GetField("Image") == plot:
                        geom = ft.GetGeometryRef()
                        extent = geom.GetEnvelope()
                    ft = lr.GetNextFeature()
                ft = None
                lr = None
                x_min, x_max, y_min, y_max = extent
            t_crop = t[:len(t)-4] + "_CROP.tif"
            tif = gdal.Open(t)
            tif_crop = gdal.Warp(t_crop, tif,
                                 outputBounds = (x_min, y_min, x_max, y_max))
            tif_crop.SetProjection(tif.GetProjection())
            tif = None
            tif_crop = None
            from osgeo import gdalconst
            tif_crop = gdal.Open(t_crop, gdalconst.GA_ReadOnly)#r"/vsimem/clip.tif"
            m = t[:len(t)-4] + "_MASK.tif"
            x_res, y_res = tif_crop.RasterXSize, tif_crop.RasterYSize
            geo_tf = tif_crop.GetGeoTransform()
            pxw = geo_tf[1]
            mask = gdal.GetDriverByName("GTiff") \
                .Create(m, x_res, y_res, 1, gdal.GDT_Byte)
            mask.SetGeoTransform((x_min, pxw, 0, y_min, 0, pxw))
            mask.SetProjection(tif_crop.GetProjection())
            CopyDatasetInfo(tif_crop, mask)
            band = mask.GetRasterBand(1)
            band.Fill(NoDataValue)
            gdal.RasterizeLayer(mask, [1], layer, options=["ATTRIBUTE=Class"])
            #band.SetNoDataValue(NoDataValue)
            band.FlushCache()
            if mark_bad_data:
                band2 = mask.GetRasterBand(2)
                band2.Fill(0)
                gdal.RasterizeLayer(mask, [2], layer, \
                                    options = ["ATTRIBUTE=Bad_data"])
                band2 = None
            band = None
            mask = None
            layer = None
            shp = None
        else:
            t_crop = os.path.join(dir_omk(plot, type_ext = "_CROP")[0])
            m = os.path.join(dir_omk(plot, type_ext = "_MASK")[0])
        ### check if training tiles were generated previously and are up-to-date
        path_to_tiles = list(pathlib.Path(dir_tls(mdate = date, dset = set_y)) \
            .glob("**/" + plot + "*_y." + yf))
        if len(path_to_tiles) >= 1:
            im = Image.open(path_to_tiles[0])
            w, h = im.size
            im.close()
        else:
            w = 0
        keep_tiles = True if (just_add and w == imgc) else False
        if check_version(file = list(pathlib.Path(dir_tls(mdate = date,
                                                       dset = set_y)) \
                                  .glob("**/" + plot + "*_y." + yf)),
                      derived_from = list(pathlib.Path(dir_omk(mdate = date)).\
                                          glob("**/*" + plot + "_MASK.tif"))
                      ) in (None, False) and not keep_tiles:
            print("Creating tiles", set_X, set_y, "for", plot)
            ## create tiles
            ### blacken pixels not assigned to any class
            if no_data_class:
                tif = blacken(data = t_crop, mask = m, NoDataVal = NoDataValue)
            else:
                tif = gdal.Open(t_crop)
            mask = gdal.Open(m)
            r = mask.RasterYSize
            c = mask.RasterXSize
            geo_tf = mask.GetGeoTransform()
            Xmin = geo_tf[0]
            Ymax = geo_tf[3]
            res = geo_tf[1]
            ytiles = math.floor(r/imgr)
            xtiles = math.floor(c/imgc)
            ypx = ytiles*int(imgr)
            xpx = xtiles*int(imgc)
            ### offset the raster so the unused area is equally distributed at all
            ### sides of the rectangle
            yoffset = math.floor(0.5*(r - ypx))
            xoffset = math.floor(0.5*(c - xpx))
            ymax = Ymax - yoffset * res
            xmin = Xmin + xoffset * res
            ycrds = [ymax - res*imgr*m for m in range(ytiles + 1)]
            xcrds = [xmin + res*imgc*n for n in range(xtiles + 1)]
            os.makedirs(dir_tls(mdate = date, dset = set_X), exist_ok = True)
            os.makedirs(dir_tls(mdate = date, dset = set_y), exist_ok = True)
            ### delete old files
            if delete_old_tiles:
                old_tiles = []
                for root, dirs, files in os.walk(dir_tls(mdate = date,\
                                                         dset = set_X)):
                    for file in files:
                        if file.endswith(plot + "*." + xf):
                            old_tiles.append(os.path.join(root, file))
                for root, dirs, files in os.walk(dir_tls(mdate = date,\
                                                         dset = set_y)):
                    for file in files:
                        if file.endswith(plot + "*." + yf):
                            old_tiles.append(os.path.join(root, file))
                for file in old_tiles:
                     os.remove(os.path.join(file))
            delete_old_tiles = False
            counter = 0
            for i in range(ytiles):
                for j in range(xtiles):
                    ymin = ycrds[i+1]
                    ymax = ycrds[i]
                    xmin = xcrds[j]
                    xmax = xcrds[j+1]
                    x = str(counter).zfill(8)
                    fntif = dir_tls(plot_id = plot,
                                    mdate = date, dset = set_X) + x + "_X."+xf
                    fnmsk = dir_tls(plot_id = plot,
                                    mdate = date, dset = set_y) + x + "_y."+yf
                    # https://gdal.org/python/osgeo.gdal-module.html#TranslateOptions
                    os.environ["GDAL_PAM_ENABLED"] = "NO"# suppress .aux files
                    if xf == "tif":# clip and save as tif
                        gdal.Warp(fntif, tif,
                                  outputBounds = (xmin, ymin, xmax, ymax))
                    elif xf == "png":# clip and save as png
                        out_ds = gdal.Translate(fntif, tif,
                                       projWin = [xmin, ymax, xmax, ymin])
                    if yf == "tif":# clip and save as tif
                        out_ds = gdal.Warp(fnmsk, mask,
                                  outputBounds = (xmin, ymin, xmax, ymax))
                    elif yf == "png":# clip and save as png
                        out_ds = gdal.Translate(fnmsk, mask,
                                       projWin = [xmin, ymax, xmax, ymin])
                    counter = counter + 1
            tif = None
            mask = None
            out_ds = None
    ### delete all-black tiles
    y_tiles = list(pathlib.Path(dir_tls(mdate = date, dset = set_y)) \
                   .glob("**/*." + yf))
    if no_data_class:
        all_black = []
        for i in y_tiles:
            tile = gdal.Open(os.path.join(i))
            band = tile.GetRasterBand(1)
            vals = np.unique(band.ReadAsArray())
            band = None
            tile = None
            if min(vals) == NoDataValue and max(vals) == NoDataValue:
                all_black.append(i)
        for i in all_black:
            corr_X = os.path.join(i.parents[2], "X", "0",
                                  str(i.name).replace("y", "X"))
            os.remove(corr_X)
            os.remove(os.path.join(i))

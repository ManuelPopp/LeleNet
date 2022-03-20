#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 17 17:56:34 2021

@author: Manuel
Loading data and preparing the training tiles.
"""
__author__ = "Manuel R. Popp"

## select plots to use
use_plots = ["B1_6_0003", "B1_6_0023", "B1_6_0063", "B1_6_0078", "B1_6_0086",\
             "B1_6_0119", "B1_6_0136", "B1_6_0140"]
year = "03_2021"
no_data_class = False

## AcrGIS Online log-in
if not os.path.exists(os.path.join(wd, "gis", "pw.txt")):
    from cryptography.fernet import Fernet
    import urllib
    url = "https://www.dropbox.com/s/vcuezzq01drsx9m/enc.txt?dl=1"
    file = urllib.request.urlopen(url)
    for line in file:
        l = line.decode("utf-8")
        b = bytes(l, "utf-8")
        key = None
        fernet = Fernet(key)
    login = eval(fernet.decrypt(b).decode())
    key = Fernet.generate_key()
    enc = fernet.encrypt(str(login).encode())
else:
    login = open(os.path.join(wd, "gis", "pw.txt"), "r").readlines()

import arcgis, math
from arcgis.gis import GIS
gis = GIS(None, login[0].rstrip(), login[1], verify_cert = False)
def dir_shp(plot_id = None, myear = None):
    if myear == None:
        if plot_id == None:
            return(dir_dat("shp"))
        else:
            print("Option myear missing but required.")
    else:
        if plot_id == None:
            return(os.path.join(dir_dat("shp"), myear))
        else:
            os.makedirs(os.path.join(dir_dat("shp"), myear, plot_id),
                        exist_ok = True)
            return(os.path.join(dir_dat("shp"), myear, plot_id))
del login

def downloadShapefiles(plot_id, path, dateTime = None):
    try:
        ## Search items by username
        cont = gis.content.search("owner:{0}".format("manuel.popp_KIT"))
        for i in range(len(cont)):
            shp = cont[i]
            if plot_id in shp.title and shp.type == "Feature Service":
                if len(list(pathlib.Path(path). \
                             glob("**/*" + plot_id + ".shp"))) < 1:
                    update = True
                elif not datetime == None:
                    item = gis.content.get(shp.id)
                    data_modfd = max(
                        j.properties.editingInfo.lastEditDate
                        for j
                        in item.layers + item.tables
                        )
                    update = data_modfd/1000 > dateTime
                    #somehow, the ArcGIS timestamp has to be divided by 1000
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

def encode_classes(path, classes):
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
        c = classes.index(spec)
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
    else:
        if len(derived_from) < 1:
            print("Please pass a valid path to the derived_from argument.")
        else:
            file_mod = file[0].stat().st_mtime
            from_mod = derived_from[0].stat().st_mtime
            return file_mod > from_mod

## download image extents shapefile
if not os.path.isfile(os.path.join(dir_shp("Extents", myear = year),
                               "Extents.shp")):
    last_modified = None
else:
    last_modified = pathlib.Path(dir_shp("Extents", myear = year),
                               "Extents.shp").stat().st_mtime
downloadShapefiles(plot_id = "Extents",
                           path = dir_shp("Extents", myear = year),
                           dateTime = last_modified)
Ext = os.path.join(dir_shp("Extents", myear = year), "Extents.shp")
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

# run for each image or plot
for plot in use_plots:
    ### update shapefile if required
    if len(list(pathlib.Path(dir_shp(plot, myear = year)) \
                .glob("**/*" + plot + ".shp"))) < 1:
        last_modified = None
    else:
        last_modified = list(pathlib.Path(dir_shp(plot, myear = year)). \
                             glob("**/*" + plot + ".shp"))[0].stat().st_mtime
    downloadShapefiles(plot_id = plot,
                           path = dir_shp(plot, myear = year),
                           dateTime = last_modified)
## get class IDs
classes_decoded = get_classes(dir_shp(myear = year), plots = use_plots)
# NoDataValue = Class "bare soil"
NoDataValue = len(classes_decoded)
classes = range(len(classes_decoded))

save_var(variables = [classes, classes_decoded, NoDataValue, no_data_class],
         name = "ClassIDs")

for plot in use_plots:
    shp_path = list(pathlib.Path(dir_shp(plot, myear = year)) \
                    .glob("**/*" + plot + ".shp"))[0]
    encode_classes(path = os.path.join(shp_path), classes = classes_decoded)
    ### tif and shp path
    t = os.path.join(dir_omk(plot)[0])
    plib = pathlib.Path(dir_shp(plot, myear = year))
    ### check if shapefile was transformed to UTM
    if check_version(file = list(pathlib.Path(dir_shp(plot, myear = year)). \
                              glob("**/*_UTM.shp")),
                  derived_from = list(
                      pathlib.Path(dir_shp(plot, myear = year)). \
                              glob("**/*" + plot + ".shp"))) in (None, False):
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
            ds = gdal.VectorTranslate(sUTM, srcDS, format = "ESRI Shapefile",
                                      dstSRS = crs1, reproject = True)
            ds = None
            srcDS = None
            ### write .prj file
            with open(f"{os.path.splitext(sUTM)[0]}.prj", "w") as f:
                f.write(crs1)
        else:
            raise Exception("Raster is missing (for layout reference): " + t)
    if check_version(file = dir_omk(plot, type_ext = "_MASK"),
                     derived_from = list(pathlib.Path(dir_shp(plot,
                                                              myear = year)). \
                              glob("**/*_UTM.shp"))) in (None, False):
        ## create mask
        sUTM = os.path.join(list(plib.glob("**/*_UTM.shp"))[0])
        drv = ogr.GetDriverByName("ESRI Shapefile")
        shp = drv.Open(sUTM)
        layer = shp.GetLayer()
        extent_from_shapefile = False
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
        x_res = tif_crop.RasterXSize
        y_res = tif_crop.RasterYSize
        geo_tf = tif_crop.GetGeoTransform()
        pxw = geo_tf[1]
        mask = gdal.GetDriverByName("GTiff") \
            .Create(m, x_res, y_res, 1, gdal.GDT_Byte)
        mask.SetGeoTransform((x_min, pxw, 0, y_min, 0, pxw))
        mask.SetProjection(tif_crop.GetProjection())
        CopyDatasetInfo(tif_crop, mask)
        band = mask.GetRasterBand(1)
        band.Fill(NoDataValue)
        gdal.RasterizeLayer(mask, [1], layer, options = ["ATTRIBUTE=Class"])
        #band.SetNoDataValue(NoDataValue)
        band.FlushCache()
        band = None
        mask = None
        layer = None
        shp = None
    else:
        t_crop = os.path.join(dir_omk(plot, type_ext = "_CROP")[0])
        m = os.path.join(dir_omk(plot, type_ext = "_MASK")[0])
    ### check if training tiles were generated previously and are up-to-date
    if check_version(file = list(pathlib.Path(dir_tls(plot_id = plot,
                                                   myear = year)) \
                              .glob("**/*_y." + yf)),
                  derived_from = list(pathlib.Path(dir_omk(myear = year)). \
                                      glob("**/*" + plot + "_MASK.tif"))
                  ) in (None, False):
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
        ### walk North, since we're in the southern hemisphere
        ycrds = [ymax - res*imgr*m for m in range(ytiles+1)]
        xcrds = [xmin + res*imgc*n for n in range(xtiles+1)]
        os.makedirs(dir_tls(myear = year, dset = "X"), exist_ok = True)
        os.makedirs(dir_tls(myear = year, dset = "y"), exist_ok = True)
        ### delete old files
        old_X_tiles = list(pathlib.Path(dir_tls(myear = year, dset = "X")) \
                           .glob("**/" + plot + "*"))
        old_y_tiles = list(pathlib.Path(dir_tls(myear = year, dset = "y")) \
                           .glob("**/" + plot + "*"))
        for file in old_X_tiles:
             os.remove(os.path.join(file))
        for file in old_y_tiles:
             os.remove(os.path.join(file))
        counter = 0
        for i in range(ytiles):
            for j in range(xtiles):
                ymin = ycrds[i+1]
                ymax = ycrds[i]
                xmin = xcrds[j]
                xmax = xcrds[j+1]
                x = str(counter).zfill(8)
                fntif = dir_tls(plot_id = plot,
                                myear = year, dset = "X") + x + "_X." + xf
                fnmsk = dir_tls(plot_id = plot,
                                myear = year, dset = "y") + x + "_y." + yf
                # https://gdal.org/python/osgeo.gdal-module.html#TranslateOptions
                os.environ["GDAL_PAM_ENABLED"] = "NO"# suppress .aux files
                if xf == "tif":# clip and save as tif
                    gdal.Warp(fntif, tif,
                              outputBounds = (xmin, ymin, xmax, ymax))
                elif xf == "png":# clip and save as png
                    gdal.Translate(fntif, tif,
                                   projWin = [xmin, ymax, xmax, ymin])
                if yf == "tif":# clip and save as tif
                    gdal.Warp(fnmsk, mask,
                              outputBounds = (xmin, ymin, xmax, ymax))
                elif yf == "png":# clip and save as png
                    gdal.Translate(fnmsk, mask,
                                   projWin = [xmin, ymax, xmax, ymin])
                counter = counter + 1
        tif = None
        mask = None
### delete all-black tiles
y_tiles = list(pathlib.Path(dir_tls(myear = year, dset = "y")) \
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

## split in training and validation set
### set share of validation tiles
tst_share = 0.1
val_share = 0.25
N_tot = len(list(pathlib.Path(dir_tls(myear = year, dset = "X")) \
                         .glob("**/*." + xf)))
### check if tiles are up-to-date
tiles_Xmfd = pathlib.Path(dir_tls(myear = year, dset = "X")).stat() \
    .st_mtime
### initialize, if files are missing
import shutil
for SET in ["tst", "val"]:
    if not os.path.exists(dir_tls(myear = year, dset = "X_" + SET)):
        copy_new = True
    else:
        ##### delete old, if train set is newer than tst/val set
        if tiles_Xmfd > pathlib.Path(dir_tls(myear = year,
                                        dset = "X_" + SET)).stat().st_mtime:
            copy_new = True
            shutil.rmtree(dir_tls(myear = year, dset = "X_" + SET))
            shutil.rmtree(dir_tls(myear = year, dset = "y_" + SET))
    if copy_new:
        os.makedirs(dir_tls(myear = year, dset = "X_" + SET), exist_ok = True)
        os.makedirs(dir_tls(myear = year, dset = "y_" + SET), exist_ok = True)
        N_totX = len(list(pathlib.Path(dir_tls(myear = year, dset = "X")) \
                         .glob("**/*." + xf)))
        N_toty = len(list(pathlib.Path(dir_tls(myear = year, dset = "y")) \
                         .glob("**/*." + yf)))
        if not N_totX == N_toty:
            raise Exception("Number of X tiles does not match y tiles.")
        else:
            N_tot = N_totX
            N_set = int(round(N_tot*globals()[SET + "_share"]))
            del N_totX, N_toty
        import random
        set_indices = random.sample(range(N_tot), N_set)
        X_tiles = list(pathlib.Path(dir_tls(myear = year, dset = "X")) \
                         .glob("**/*." + xf))
        X_set_tiles = [X_tiles[i] for i in set_indices]
        y_tiles = list(pathlib.Path(dir_tls(myear = year, dset = "y")) \
                         .glob("**/*." + yf))
        y_set_tiles = [y_tiles[i] for i in set_indices]
        for X_tile in X_set_tiles:
            shutil.move(os.path.join(X_tile),
                        dir_tls(myear = year, dset = "X_" + SET))
        for y_tile in y_set_tiles:
            shutil.move(os.path.join(y_tile),
                        dir_tls(myear = year, dset = "y_" + SET))

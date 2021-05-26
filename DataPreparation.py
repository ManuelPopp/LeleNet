#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 17 17:56:34 2021

@author: Manuel
Loading data and preparing the training tiles.
"""
__author__ = "Manuel R. Popp"

### select plots to use
use_plots = ["B3_4"]
year = "03_2021"
no_data_class = True

#### AcrGIS Online log-in
if not os.path.exists(os.path.join(wd, "gis", "pw.txt")):
    from cryptography.fernet import Fernet
    import urllib
    url = "https://www.dropbox.com/s/vcuezzq01drsx9m/enc.txt?dl=1"
    file = urllib.request.urlopen(url)
    for line in file:
        l = line.decode("utf-8")
        b = bytes(l, "utf-8")
        key = b"5v-V22Z8KfJjs6XlcMaO1OxDhx5mXxC0YMuKTI67pSs="
        fernet = Fernet(key)
    login = eval(fernet.decrypt(b).decode())
    key = Fernet.generate_key()
    enc = fernet.encrypt(str(login).encode())
else:
    login = open(os.path.join(wd, "gis", "pw.txt"), "r").readlines()

import arcgis
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
        # Search items by username
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

def get_classes(path):
    from osgeo import ogr
    drv = ogr.GetDriverByName("ESRI Shapefile")
    # read existing classes
    files = list(pathlib.Path(path).glob("**/*.shp"))
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
            class_list.append(feature.GetField("Species"))
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
    # create new field
    dataSource = drv.Open(path, 1)
    FDef = ogr.FieldDefn("Class", ogr.OFTInteger)
    layer = dataSource.GetLayer()
    exists = layer.GetLayerDefn().GetFieldIndex("Class") >= 0
    if not exists:
        layer.CreateField(FDef)
    feature = layer.GetNextFeature()
    while feature:
        spec = feature.GetField("Species")
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

### get data and create masks and tiles
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
    # expects pathlib path list
    if len(file) < 1:
        return None
    else:
        if len(derived_from) < 1:
            print("Please pass a valid path to the derived_from argument.")
        else:
            file_mod = file[0].stat().st_mtime
            from_mod = derived_from[0].stat().st_mtime
            return file_mod > from_mod

for plot in use_plots:
    # update shapefile if required
    if len(list(pathlib.Path(dir_shp(plot, myear = year)) \
                .glob("**/*" + plot + ".shp"))) < 1:
        last_modified = None
    else:
        last_modified = list(pathlib.Path(dir_shp(plot, myear = year)). \
                             glob("**/*" + plot + ".shp"))[0].stat().st_mtime
    downloadShapefiles(plot_id = plot,
                           path = dir_shp(plot, myear = year),
                           dateTime = last_modified)
    shp_path = list(pathlib.Path(dir_shp(plot, myear = year)) \
                    .glob("**/*" + plot + ".shp"))[0]
# get class IDs
classes_decoded = get_classes(dir_shp(myear = year))
classes = range(len(classes_decoded))
NoDataValue = max(classes) + 1
save_var(variables = [classes, classes_decoded, NoDataValue],
         name = "ClassIDs")

for plot in use_plots:
    encode_classes(path = os.path.join(shp_path), classes = classes_decoded)
    # tif and shp path
    t = os.path.join(dir_omk(plot)[0])
    plib = pathlib.Path(dir_shp(plot, myear = year))
    # check if shapefile was transformed to UTM
    if check_version(file = list(pathlib.Path(dir_shp(plot, myear = year)). \
                              glob("**/*_UTM.shp")),
                  derived_from = list(
                      pathlib.Path(dir_shp(plot, myear = year)). \
                              glob("**/*" + plot + ".shp"))) in (None, False):
        if os.path.exists(t):
            # open orthomosaic
            tif = gdal.Open(t)
            crs1 = tif.GetSpatialRef()#.GetProjection()
            tif = None
            drv = ogr.GetDriverByName("ESRI Shapefile")
            # open shapefile
            s = os.path.join(list(plib.glob("**/*.shp"))[0])
            sUTM = s[:len(s)-4] + "_UTM.shp"
            LayerName = pathlib.Path(s).name
            if os.path.exists(sUTM):
                drv.DeleteDataSource(sUTM)
            shp = drv.Open(s)
            layer = shp.GetLayer()
            crs0 = layer.GetSpatialRef()
            # transform to similar crs
            coordTrans = osr.CoordinateTransformation(crs0, crs1)
            # create output shapefile
            outSHP = drv.CreateDataSource(sUTM)
            outLayer = outSHP.CreateLayer(LayerName[:len(LayerName)-4],
                                          geom_type = ogr.wkbMultiPolygon)
            lyrdef = layer.GetLayerDefn()
            for i in range(0, lyrdef.GetFieldCount()):
                fieldDefn = lyrdef.GetFieldDefn(i)
                outLayer.CreateField(fieldDefn)
            outdef = outLayer.GetLayerDefn()
            feature = layer.GetNextFeature()
            while feature:
                geom = feature.GetGeometryRef()
                geom.Transform(coordTrans)
                outFeature = ogr.Feature(outdef)
                outFeature.SetGeometry(geom)
                for i in range(0, outdef.GetFieldCount()):
                    outFeature.SetField(outdef.GetFieldDefn(i).GetNameRef(),
                                        feature.GetField(i))
                outLayer.CreateFeature(outFeature)
                outFeature = None
                feature = layer.GetNextFeature()
            layer = None
            shp = None
            outSHP = None
        else:
            raise Exception("Raster is missing (for layout reference): " + t)
    if check_version(file = dir_omk(plot, type_ext = "_MASK"),
                     derived_from = list(pathlib.Path(dir_shp(plot,
                                                              myear = year)). \
                              glob("**/*_UTM.shp"))) in (None, False):
        # create mask
        sUTM = os.path.join(list(plib.glob("**/*_UTM.shp"))[0])
        drv = ogr.GetDriverByName("ESRI Shapefile")
        shp = drv.Open(sUTM)
        layer = shp.GetLayer()
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
        t_crop = dir_omk(plot, type_ext = "_CROP")
        m = dir_omk(plot, type_ext = "_MASK")
    # check if training tiles were generated
    if check_version(file = list(pathlib.Path(dir_tls(plot_id = plot,
                                                   myear = year)) \
                              .glob("**/*_y.tif")),
                  derived_from = list(pathlib.Path(dir_omk(myear = year)). \
                                      glob("**/*MASK.tif"))
                  ) in (None, False):
        ### create tiles
        # blacken pixels not assigned to any class
        if no_data_class:
            tif = blacken(data = t_crop, mask = m, NoDataVal = NoDataValue)
        else:
            tif = gdal.Open(t_crop)
        mask = gdal.Open(m)
        r = mask.RasterYSize
        c = mask.RasterXSize
        geo_tf = mask.GetGeoTransform()
        xmin = geo_tf[0]
        ymax = geo_tf[3]
        res = geo_tf[1]
        ytiles = int(r/imgr)
        xtiles = int(c/imgc)
        ypx = ytiles*int(imgr)
        xpx = xtiles*int(imgc)
        # walk North, since we're in the southern hemisphere
        ycrds = [ymax - res*imgr*m for m in range(ytiles+1)]
        xcrds = [xmin + res*imgc*n for n in range(xtiles+1)]
        os.makedirs(dir_tls(myear = year, dset = "X"), exist_ok = True)
        os.makedirs(dir_tls(myear = year, dset = "y"), exist_ok = True)
        # delete old files
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
                                myear = year, dset = "X") + x + "_X.tif"
                fnmsk = dir_tls(plot_id = plot,
                                myear = year, dset = "y") + x + "_y.tif"
                gdal.Warp(fntif, tif, outputBounds = (xmin, ymin, xmax, ymax))
                gdal.Warp(fnmsk, mask, outputBounds = (xmin, ymin, xmax, ymax))
                counter = counter + 1
        tif = None
        mask = None
# delete all-black tiles
y_tiles = list(pathlib.Path(dir_tls(myear = year, dset = "y")) \
               .glob("**/*.tif"))
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

#### split in training and validation set
# set share of validation tiles
tst_share = 0.1
val_share = 0.25
N_tot = len(list(pathlib.Path(dir_tls(myear = year, dset = "X")) \
                         .glob("**/*.tif")))
# check if tiles are up-to-date
tiles_Xmfd = pathlib.Path(dir_tls(myear = year, dset = "X")).stat() \
    .st_mtime
# initialize, if files are missing
import shutil
for SET in ["tst", "val"]:
    if not os.path.exists(dir_tls(myear = year, dset = "X_" + SET)):
        copy_new = True
    else:
        # delete old, if train set is newer than tst/val set
        if tiles_Xmfd > pathlib.Path(dir_tls(myear = year,
                                        dset = "X_" + SET)).stat().st_mtime:
            copy_new = True
            shutil.rmtree(dir_tls(myear = year, dset = "X_" + SET))
            shutil.rmtree(dir_tls(myear = year, dset = "y_" + SET))
    if copy_new:
        os.makedirs(dir_tls(myear = year, dset = "X_" + SET), exist_ok = True)
        os.makedirs(dir_tls(myear = year, dset = "y_" + SET), exist_ok = True)
        N_totX = len(list(pathlib.Path(dir_tls(myear = year, dset = "X")) \
                         .glob("**/*.tif")))
        N_toty = len(list(pathlib.Path(dir_tls(myear = year, dset = "y")) \
                         .glob("**/*.tif")))
        if not N_totX == N_toty:
            raise Exception("Number of X tiles does not match y tiles.")
        else:
            N_set = int(round(N_tot*globals()[SET + "_share"]))
            del N_totX, N_toty
        import random
        set_indices = random.sample(range(N_tot), N_set)
        X_tiles = list(pathlib.Path(dir_tls(myear = year, dset = "X")) \
                         .glob("**/*.tif"))
        X_set_tiles = [X_tiles[i] for i in set_indices]
        y_tiles = list(pathlib.Path(dir_tls(myear = year, dset = "y")) \
                         .glob("**/*.tif"))
        y_set_tiles = [y_tiles[i] for i in set_indices]
        for X_tile in X_set_tiles:
            shutil.move(os.path.join(X_tile),
                        dir_tls(myear = year, dset = "X_" + SET))
        for y_tile in y_set_tiles:
            shutil.move(os.path.join(y_tile),
                        dir_tls(myear = year, dset = "y_" + SET))
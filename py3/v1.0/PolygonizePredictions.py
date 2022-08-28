# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 01:19:05 2022

@author: Manuel
"""
__author__ = "Manuel R. Popp"

#### parse arguments-----------------------------------------------------------
import argparse
import urllib.request # in case it isn't loaded properly by arcgis

def parseArguments():
    parser = argparse.ArgumentParser()
    # Positional mandatory arguments
    parser.add_argument("pd",\
                        help = "Directory containing the predictions.",\
                            type = str)
    # Optional arguments
    parser.add_argument("-wd", "--wd",\
                        help = "Alternative working directory.", type = str,\
                            default = "")
    parser.add_argument("-dd", "--dd",\
                        help = "Alternative data directory.", type = str,\
                            default = None)
    parser.add_argument("-sx", "--sx",\
                        help = "Suffix of prediction images.", type = str,\
                            default = "_PRED")
    parser.add_argument("-ndv", "--ndv",\
                        help = "No data value.", type = int,\
                            default = 255)
    parser.add_argument("-pf", "--pf",\
                        help = "ArcGIS Online project folder.", type = str,\
                            default = "LeleNet")
    # Parse arguments
    args = parser.parse_args()
    return args
if __name__ == "__main__":
    # Parse the arguments
    args = parseArguments()

wd = args.wd
dd = args.dd
pdir = args.pd
pdir = pdir.replace("\\", "/")
suffix = args.sx
ndv = args.ndv
proj_folder = args.pf

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
import os
import re
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

dd = wd if dd is None else dd
pdir = os.path.join(wd, "dat") if pdir is None else pdir

def dir_dat(*args):
    if len(args) == 0:
        return os.path.join(dd, "dat")
    else:
        dat_id = "/".join(args)
        dat_id = re.split("[,/+ ]", dat_id)
        return os.path.join(dd, "dat", *dat_id)

def dir_xls(*args):
    xls_dir = dir_dat("xls")
    if len(args) == 0:
        return xls_dir
    else:
        xls_id = "/".join(args)
        xls_id = re.split("[,/+ ]", xls_id)
        return os.path.join(xls_dir, *xls_id)

def dir_out(*args):
    out_dir = os.path.join(wd, "out")
    if len(args) == 0:
        return out_dir
    else:
        out_id = "/".join(args)
        out_id = re.split("[,/+ ]", out_id)
        return os.path.join(out_dir, *out_id)

### get list-------------------------------------------------------------------
import pandas as pd
polyg_xlsx = pd.read_excel(dir_xls("Datasets.xlsx"),
                         sheet_name = "Polygonize", header = 0)
polyg_list = polyg_xlsx[polyg_xlsx["Polygonize"] == "Yes"]["Name"].tolist()
class_dict = pd.read_excel(dir_xls("SpeciesList.xlsx"),
                         sheet_name = "Dictionary", header = 0)

### polygonize-----------------------------------------------------------------
import numpy as np
import tempfile
from osgeo import gdal, ogr, osr

def decode_classes(path, custom_dict = None):
    drv = ogr.GetDriverByName("ESRI Shapefile")
    ## create new field
    dataSource = drv.Open(path, 1)
    FDef0 = ogr.FieldDefn("SpeciesID", ogr.OFTInteger)
    layer = dataSource.GetLayer()
    exists = layer.GetLayerDefn().GetFieldIndex("SpeciesID") >= 0
    if not exists:
        layer.CreateField(FDef0)
    FDef1 = ogr.FieldDefn("Bad_data", ogr.OFTInteger)
    layer = dataSource.GetLayer()
    exists = layer.GetLayerDefn().GetFieldIndex("SpeciesID") >= 0
    if not exists:
        layer.CreateField(FDef1)
    max_class = max(custom_dict.Class)
    feature = layer.GetNextFeature()
    while feature:
        clss = feature.GetField("Class")
        matches = custom_dict.loc[custom_dict.Class == clss, "SpeciesID"] if \
            clss <= max_class else []
        if len(matches) == 0:
            s = ndv
        elif len(matches) == 1:
            s = matches.item()
        else:
            s = matches.iloc[0]
        s = int(s)
        feature.SetField("SpeciesID", s)
        feature.SetField("Bad_data", 0)
        layer.SetFeature(feature)
        feature = layer.GetNextFeature()
    dataSource.SyncToDisk()
    feature = None
    layer = None
    dataSource = None

def simplify_poly(path, tolerance = 0.005):
    shp = ogr.Open(path, 1)
    layer = shp.GetLayer()
    feature = layer.GetNextFeature()
    while feature:
        geom = feature.geometry()
        simple = geom.Simplify(tolerance)
        feature.SetGeometry(simple)
        layer.SetFeature(feature)
        feature = layer.GetNextFeature()
    shp.SyncToDisk()
    layer = None
    shp = None

## AcrGIS Online log-in
import shutil
import arcgis
from arcgis.gis import GIS
gis = GIS("pro")
tmp_dir = tempfile.mkdtemp()

print("Reading from directory " + pdir)
print(polyg_list)

for img in polyg_list:
    # create temporary shapefile
    print("Load raster: " + img + suffix + ".tif")
    prd_path = os.path.join(pdir, img + suffix + ".tif")
    shp_dir = os.path.join(tmp_dir, img)
    os.makedirs(shp_dir, exist_ok = True)
    prd = gdal.Open(prd_path, gdal.GA_ReadOnly)
    band = prd.GetRasterBand(1)
    shp_layername = img
    shp_classField = "Class"
    shp_speciesField = "SpeciesID"
    
    print("Create temporary shapefile at " + shp_dir)
    drv = ogr.GetDriverByName("ESRI Shapefile")
    shp = drv.CreateDataSource(os.path.join(shp_dir, img + ".shp"))
    spatial_ref = osr.SpatialReference()
    spatial_ref.ImportFromEPSG(32735)
    layer_class = shp.CreateLayer(shp_layername, srs = spatial_ref)
    field0 = ogr.FieldDefn(shp_classField, ogr.OFTInteger)
    layer_class.CreateField(field0)
    f = layer_class.GetLayerDefn().GetFieldIndex(shp_classField)
    print("Polygonising raster...")
    gdal.Polygonize(band, None, layer_class, f, [], callback = None)
    layer_class = None
    shp = None
    prd = None
    
    # set SpeciesID values
    print("Decode classes")
    decode_classes(os.path.join(shp_dir, img + ".shp"), \
                   custom_dict = class_dict)
    
    # simplify polygons
    print("Simplify polygons")
    simplify_poly(os.path.join(shp_dir, img + ".shp"))

    # upload to ArcGIS Online
    print("Zip shapefile")
    shutil.make_archive(shp_dir, format = "zip", root_dir = shp_dir)
    print("Uploading Feature Layer to ArcGIS Online...")
    shp_agol = gis.content.add({"title": img,
                                "tags": "Predicted",
                                "type": "Shapefile"}, shp_dir + ".zip")
    print("Publishing Feature Layer " + img)
    fs_agol = shp_agol.publish({"name": img}, build_initial_cache = True, \
                               overwrite = True)
    print("Moving Feature layer to " + proj_folder)
    fs_agol.move(folder = proj_folder)
    print("Cleaning up...")
    shp_agol.delete()
    shp_agol = None
    fs_agol = None
    
    # delete temporary files
    shutil.rmtree(shp_dir)
    os.remove(shp_dir + ".zip")
print("Process finished at " + datetime.datetime.utcnow())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 10:25:51 2022

@author: Manuel
Predict classes
Run from terminal: python3 LeleNet_prd.py /path/to/orthomosaic /path/to/output /path/to/model
"""
__author__ = "Manuel R. Popp"

#### parse arguments-----------------------------------------------------------
# Import arguments
import argparse, os, pathlib, pickle

def parseArguments():
    parser = argparse.ArgumentParser()
    # Positional mandatory arguments
    parser.add_argument("input", help = "Path to input orthomosaic" + \
                        "(3 band GeoTiff).", type = str)
    parser.add_argument("output", help = "Path to input output file" + \
                        "(grayscale GeoTiff).", type = str)
    parser.add_argument("model", help = "Path to trained model.", \
                        type = str)
    # Parse arguments
    args = parser.parse_args()
    return args
if __name__ == "__main__":
    # Parse the arguments
    args = parseArguments()

path_omk = args.input
path_prediction = args.output
path_model = args.model
no_data_value = 0

if os.path.isfile(path_model):
    path_model = pathlib.Path(path_model).parent.absolute()
else:
    path_model = pathlib.Path(path_model)

#### load model and custom objects---------------------------------------------
import tensorflow as tf
from tensorflow import keras as ks

# Load custom metrics
path_custom_metrics = path_model.glob("*metric*.pkl")
path_custom_metrics = [p for p in path_custom_metrics][0]

with open(path_custom_metrics, "rb") as f:
    met = pickle.load(f)

co = {"MulticlassMeanIoU": met} if met.__name__ == "Multi_MeanIoU" else \
    {"F1": met}
model = ks.models.load_model(path_model, \
                             custom_objects = co)

input_shape = model.layers[0].input_shape

#### use the model to predict classes------------------------------------------
from osgeo import gdal, osr
import numpy as np
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
from osgeo.gdal_array import CopyDatasetInfo, BandReadAsArray

if os.path.exists(path_omk):
    # get GeoTIFF info
    omk = gdal.Open(path_omk)
    n_cols = omk.RasterXSize
    n_rows = omk.RasterYSize
    GeoTransf = omk.GetGeoTransform()
    proj = osr.SpatialReference()
    proj.ImportFromWkt(omk.GetProjectionRef())
    omk = None
    # create empty GeoTIFF
    DataType = gdal.GDT_Byte
    driver = gdal.GetDriverByName("GTiff")
    outDS = driver.Create(path_prediction, n_cols, n_rows, 1, DataType)
    outDS.SetGeoTransform(GeoTransf)
    outDS.SetProjection(proj.ExportToWkt())
    band = outDS.GetRasterBand(1)
    band.SetNoDataValue(no_data_value)
    band = None
    outDS = None
    
    # calculate tile information
    imgc = input_shape[0][1]
    imgr = input_shape[0][2]
    n_tiles_x = n_cols // imgc
    remainder_x = n_cols % imgc
    n_tiles_y = n_rows // imgr
    remainder_y = n_rows % imgr
    
    n_tiles_x += 1 if remainder_x != 0 else n_tiles_x
    n_tiles_y += 1 if remainder_y != 0 else n_tiles_y
    
    # predict output
    omk = gdal.Open(path_omk)
    outDS = gdal.Open(path_prediction, gdal.GA_Update)
    outBAND = outDS.GetRasterBand(1)
    
    for i in range(n_tiles_y):
        for j in range(n_tiles_x):
            xmin = j * imgc if j < (n_tiles_x - 1) else (j - 1) * imgc + \
                remainder_x
            ymin = i * imgr if i < (n_tiles_y - 1) else (i - 1) * imgr + \
                remainder_y
            window = omk.ReadAsArray(xmin, ymin, imgc, imgr)
            window = window / 255.0
            window = np.transpose(window, (1, 2, 0))
            
            y = model.predict(np.expand_dims(window, axis = 0))
            pred_mask = tf.argmax(y, axis = -1)
            #pred_mask = tf.expand_dims(pred_mask, axis = -1)
            pred_mask = pred_mask.astype("uint8")
            pred_mask = np.squeeze(pred_mask, axis = 0)
            #pred_mask = np.squeeze(pred_mask, axis = 2)
            outBAND.WriteArray(pred_mask, xmin, ymin)
            outBAND.FlushCache()
    omk = None
    outBAND = None
    outDS = None
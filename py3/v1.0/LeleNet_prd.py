#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 10:25:51 2022

@author: Manuel
Predict classes using trained model
Run from terminal: python3 LeleNet_prd.py /path/to/orthomosaic /path/to/output /path/to/model
"""
__author__ = "Manuel R. Popp"

#### parse arguments-----------------------------------------------------------
# Import arguments
import argparse, os, pathlib, pickle, tempfile, shutil

def parseArguments():
    parser = argparse.ArgumentParser()
    # Positional mandatory arguments
    parser.add_argument("input", help = "Path to input orthomosaic" + \
                        "(3 band GeoTiff).", type = str)
    parser.add_argument("output", help = "Path to input output file" + \
                        "(grayscale GeoTiff).", type = str)
    parser.add_argument("model", help = "Path to trained model.", \
                        type = str)
    parser.add_argument("-rol", "--rol", help = "Row overlap.", \
                        type = float, default = None)
    parser.add_argument("-col", "--col", help = "Column overlap.", \
                        type = float, default = None)
    parser.add_argument("-ndv", "--ndv", help = "No data value.", \
                        type = int, default = 255)
    # Note: Tested successfully with DeepLabv3+, NVIDIA GeForce GTX 970 (4 GB),
    # and 512 pixel tile dimension. 1024 px will cause OOM with this GPU model.
    parser.add_argument("-gpu", "--gpu", help = "Enable GPU usage.", \
                        action = "store_true")
    # Parse arguments
    args = parser.parse_args()
    return args
if __name__ == "__main__":
    # Parse the arguments
    args = parseArguments()

path_omk = args.input
path_prediction = args.output
path_model = args.model
rolap = args.rol
colap = args.col
GPU = args.gpu
if type(GPU) is not bool:
    GPU = (GPU.casefold()=="true")

no_data_value = args.ndv

tmp_dir = tempfile.mkdtemp()

if os.path.isfile(path_model):
    path_model = pathlib.Path(path_model).parent.absolute()
else:
    path_model = pathlib.Path(path_model)

#### load model and custom objects---------------------------------------------
import tensorflow as tf
from tensorflow import keras as ks

# enable/disable GPU mode
print("Selected option: GPU=" + str(GPU) + ".")
if GPU:
    print("Using GPU for model prediction.")
else:
    print("Attempting to disable GPU.")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    try:
        tf.config.set_visible_devices([], "GPU")
        visible_devices = tf.config.get_visible_devices()
        for device in visible_devices:
            assert device.device_type != "GPU"
    except:
        print("Invalid device or cannot modify virtual devices " + \
              "once initialized.")

# Load custom metrics
path_custom_metrics = path_model.glob("*metric*.pkl")
path_custom_metrics = [p for p in path_custom_metrics][0]
'''
class SparseMeanIoU():
    pass
class SparseF1():
    pass

with open(path_custom_metrics, "rb") as f:
    met = pickle.load(f)
'''
class SparseMeanIoU(ks.metrics.MeanIoU):
    def __init__(self,
                 y_true = None,
                 y_pred = None,
                 num_classes = None,
                 name = "Sparse_MeanIoU",
                 dtype = None):
        super(SparseMeanIoU, self).__init__(num_classes = num_classes,
                                             name = name, dtype = dtype)
        #self.__name__ = "Sparse_MeanIoU"
    def get_config(self):
        return {"num_classes": self.num_classes, \
                "name": self.name, \
                    "dtype": self._dtype}
    '''
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    '''
    def update_state(self, y_true, y_pred, sample_weight = None):
        y_pred = tf.math.argmax(y_pred, axis = -1)
        return super().update_state(y_true, y_pred, sample_weight)
    def __getstate__(self):
        variables = {v.name: v.numpy() for v in self.variables}
        state = { \
            name: variables[var.name] \
                for name, var in self._unconditional_dependency_names.items() \
                    if isinstance(var, tf.Variable)}
        state["name"] = self.name
        state["num_classes"] = self.num_classes
        return state
    def __setstate__(self, state):
        self.__init__(name = state.pop("name"), \
            num_classes = state.pop("num_classes"))
        for name, value in state.items():
            self._unconditional_dependency_names[name].assign(value)
met = SparseMeanIoU(num_classes = 17)

co = {"SparseMeanIoU": met} if met.name == "Sparse_MeanIoU" else \
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
    
    # predict output without overlap
    if rolap is None and colap is None:
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
                pred_mask = pred_mask.astype("uint8")
                pred_mask = np.squeeze(pred_mask, axis = 0)
                outBAND.WriteArray(pred_mask, xmin, ymin)
                outBAND.FlushCache()
        omk = None
        outBAND = None
        outDS = None
    else:
        omk = gdal.Open(path_omk)
        outDS = gdal.Open(path_prediction)
        #tmpDS = driver.CreateCopy(os.path.join(tmp_dir, "tmp.tif"), outDS)
        tmpDS = gdal.GetDriverByName("MEM").CreateCopy("", outDS, 0)
        rlap = int(rolap * imgr)
        clap = int(colap * imgc)
        # iterate with row shitfs
        i = 0
        b = 1
        while rlap * i < imgr:
            roff = rlap * i
            print("Running with y offset " + str(roff), flush = True)
            # iterate with column shitfs
            j = 0
            while clap * j < imgc:
                coff = clap * j
                print("Running with x offset " + str(coff), flush = True)
                xmin = coff
                if b > 1:
                    tmpDS.AddBand()
                tmpband = tmpDS.GetRasterBand(b)
                print("Writing band " + str(b), flush = True)
                # iterate through rows and columns of the image
                while xmin < (n_cols - imgc):
                    ymin = roff
                    while ymin < (n_rows - imgr):
                        window = omk.ReadAsArray(xmin, ymin, imgc, imgr)
                        window = window / 255.0
                        window = np.transpose(window, (1, 2, 0))
                        y = model.predict(np.expand_dims(window, axis = 0))
                        pred_mask = tf.argmax(y, axis = -1)
                        pred_mask = pred_mask.astype("uint8")
                        pred_mask = np.squeeze(pred_mask, axis = 0)
                        tmpband.WriteArray(pred_mask, xmin, ymin)
                        tmpband.FlushCache()
                        ymin += imgr
                    xmin += imgc
                j += 1
                b += 1
                tmpband = None
            i += 1
        omk = None
        arrays = []
        for bm1 in range(tmpDS.RasterCount):
            b = bm1 + 1
            print("Load tmp raster band " + str(b))
            band = tmpDS.GetRasterBand(b)
            b_array = band.ReadAsArray()
            print("Append band " + str(b) + " with values " + \
                  str(np.unique(b_array)))
            arrays.append(b_array)
        from scipy import stats
        array = stats.mode(arrays).mode
        array = np.squeeze(array)
        tmpDS = None
        print("Write layer with values " + str(np.unique(array)))
        outDS = gdal.Open(path_prediction, gdal.GA_Update)
        outBAND = outDS.GetRasterBand(1)
        outBAND.WriteArray(array, 0, 0)
        outBAND.FlushCache()
        outBAND = None
        outDS = None

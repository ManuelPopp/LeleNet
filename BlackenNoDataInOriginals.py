# -*- coding: utf-8 -*-
"""
Created on Thu May 20 12:37:40 2021

@author: Manuel
"""
import numpy as np
from osgeo import gdal
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

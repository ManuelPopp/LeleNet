#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 08:19:50 2021

@author: Manuel

Execute from terminal with both of the following arguments:
1. "/path/to/files"
2. "/path/to/output"
"""
import pathlib, os, sys
import numpy as np

from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from pyproj import Proj, Transformer

# if ran from terminal with additional arguments
import argparse

def parseArguments():
    parser = argparse.ArgumentParser()
    # Positional mandatory arguments
    parser.add_argument("folder", help = "Path to the original files.",\
                        type = str)
    parser.add_argument("out_folder", help = "Output files path.",\
                        type = str)
    # Optional arguments
    parser.add_argument("-n", "--name",\
                        help = "Image prefix.",\
                            type = str, default = "")
    parser.add_argument("-reppref", "--replprefix",\
                        help = "Image prefix to replace. Standard: DJI.",\
                            type = str, default = "DJI")
    # Parse arguments
    args = parser.parse_args()
    return args

args = None
name = ""
old_prefix = ""
if __name__ == "__main__":
    # Parse the arguments
    args = parseArguments()

# else
if args is None:
    folder = "F:/Block_3/Block3_4"
    out_folder = "F:/Block_3/Block3_4_GeoTIFF"
    name = "B3_4"
else:
    folder = args.folder
    out_folder = args.out_folder
    name = args.name
    old_prefix = args.replprefix

if name == "":
    import re
    regex = re.compile("Block\d_\d")
    try:
        prefix = regex.search(str(folder)).group(0).replace("lock", "")
    except AttributeError:
        prefix = ""

file_extension = ".jpg"
inProj = "epsg:4326"
outProj = "epsg:32735"
altitude = 40
cam_angle_deg = -60
cam_angle_rad = None
ground_resolution_x = 0.011 # should be possible to calculate, but I have no time for this atm
ground_resolution_y = ground_resolution_x # just because no real value was calculated

if cam_angle_rad == None:
    cam_angle_rad = cam_angle_deg*np.pi/180
# list files
files = list(pathlib.Path(folder).glob("**/*" + file_extension))

# functions to extract location in UTM
def get_GPS_loc(filename):
    image = Image.open(filename)
    image.verify()
    exif = image._getexif()
    geotag = {}
    for (idx, tag) in TAGS.items():
        if tag == "GPSInfo":
            if idx not in exif:
                    raise ValueError("No EXIF geotagging found!")
            for (key, val) in GPSTAGS.items():
                if key in exif[idx]:
                    geotag[val] = exif[idx][key]
    return geotag

def dms_to_dd(dms, ref):
    degrees = dms[0]
    minutes = dms[1] / 60.0
    seconds = dms[2] / 3600.0
    if ref in ["S", "W"]:
        degrees = -degrees
        minutes = -minutes
        seconds = -seconds
    return round(degrees + minutes + seconds, 5)

def get_coordinates(geotag):
    lat = dms_to_dd(geotag["GPSLatitude"],
                    geotag["GPSLatitudeRef"])
    lon = dms_to_dd(geotag["GPSLongitude"],
                    geotag["GPSLongitudeRef"])
    return (lat, lon)

# extract UTM locations
coords = []
for filename in files:  
    lat, lon = get_coordinates(get_GPS_loc(filename))
    transformer = Transformer.from_crs(inProj, outProj)
    UTMY, UTMX = transformer.transform(lat, lon)
    coords.append(np.array([UTMY, UTMX]))

# extract current flight direction (drone orientation)
v_hat = []
for i in range(len(files) - 1):
    v = coords[i + 1] - coords[i]
    unit = (v / np.linalg.norm(v)) if np.linalg.norm(v) != 0. else np.array([1., 1.])   
    if not isinstance(unit[0], float):#any(unit != unit):
        unit = v_hat[i - 1]
    v_hat.append(unit)
v_hat.append(unit)# append for the last list image which has no subsequent one

def get_orientation(vec):
    N = vec[1] > 0.5*np.sqrt(2)
    O = vec[0] > 0.5*np.sqrt(2)
    S = vec[1] < -0.5*np.sqrt(2)
    W = vec[0] < -0.5*np.sqrt(2)
    return [N, O, S, W]

# rotate images
import piexif
orientation = [get_orientation(i) for i in v_hat]
for i in range(len(files)):
    file = files[i]
    if orientation[i] != [1, 0, 0, 0]:
        img = Image.open(file)
        if orientation[i] == [0, 0, 1, 0]:
            angle = 180
            EXIF = img.getexif()
            out = img.rotate(angle)#.transpose(Image.FLIP_TOP_BOTTOM)
            out.save(file, format = img.format, exif = EXIF)
        else:
            exif_dict = piexif.load(img.info["exif"])
            x = exif_dict["0th"][piexif.ImageIFD.XResolution]
            y = exif_dict["0th"][piexif.ImageIFD.YResolution]
            exif_dict["0th"][piexif.ImageIFD.XResolution] = y
            exif_dict["0th"][piexif.ImageIFD.YResolution] = x
            exif_bytes = piexif.dump(exif_dict)
            if orientation[i] == [0, 1, 0, 0]:
                angle = 270
            else:#elif orientation[i] == [0, 0, 0, 1]:
                angle = 90
            out = img.rotate(angle, expand = True)
            out.save(file, format = img.format, exif = exif_bytes)
    else:
        pass
img = None

# calculate vector between drone nadir and image center
def pos_shift(alt, cam_ang, direction):
    shift_x = np.tan(np.pi/2 + np.float64(cam_ang))*np.float64(alt)
    shift_v = np.multiply(shift_x, direction)
    return shift_v

shift_vectors = [pos_shift(alt = altitude, cam_ang = cam_angle_rad,
                           direction = i) for i in v_hat]
# calculate image centers
img_centers = [coords[i] + shift_vectors[i] for i in range(len(coords))]

def get_img_metadata(filename):
    image = Image.open(filename)
    exifdata = image._getexif()
    for tagid in exifdata:
        tagname = TAGS.get(tagid, tagid)
        value = exifdata.get(tagid)
        if tagname == "ExifImageWidth":
            img_width = value
        elif tagname == "ExifImageHeight":
            img_height = value
        #elif tagname == "FocalLength":
            #focal_length = value
    return img_width, img_height

#GSDh = altitude * sensor height / focal length * img_height
#GSDw = altitude * sensor width / focal length * img_width
from osgeo import gdal, osr
def translateIMG(path_in, path_out, gt, in_format = "GTiff"):
    ds = gdal.Open(os.path.join(path_in))
    ds = gdal.Translate(os.path.join(path_out), ds)
    ds.SetGeoTransform(gt)
    ds.SetProjection(outProj)
    ds = None

os.makedirs(out_folder, exist_ok = True)
for i in range(len(files)):
    file = files[i]
    c = img_centers[i]
    w, h = get_img_metadata(file)
    ground_w = w * ground_resolution_x
    ground_h = h * ground_resolution_y
    upleftX =  c[0] - 0.5*ground_w
    upleftY =  c[1] + 0.5*ground_h
    # (uperleftx, scalex, skewx, uperlefty, skewy, scaley)
    # Scale = size of one pixel in units of raster projection
    gt = [upleftX, ground_resolution_x, 0, upleftY, 0, -ground_resolution_y]
    p_out = os.path.join(out_folder,
                         file.name[:len(file.name)-4] + ".tif")
    if old_prefix != "":
        path_out = p_out.replace(old_prefix, prefix)
    elif prefix != "":
        path_out = os.path.join(out_folder,
                         prefix + file.name[:len(file.name)-4] + ".tif")
    else:
        path_out = p_out
    translateIMG(file, path_out, gt)

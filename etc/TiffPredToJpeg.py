# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 12:19:11 2022

@author: Manuel
"""
originals_path = "G:/Masterarbeit/MonitRaw"
predictions_path = "G:/Masterarbeit/PredRaw"
output_path = "G:/Masterarbeit/PredJPG"
MAX_CLASS_VAL = 16

import os, glob
from PIL import Image
from exif import Image as exIm
import numpy as np
os.makedirs(output_path, exist_ok = True)

files = os.listdir(predictions_path)
print("Found " + str(len(files)) + " images in " + predictions_path)

for file in files:
    f_name = "".join(file.split(".")[:-1])
    f_path_0 = os.path.join(predictions_path, file)
    f_path_1 = glob.glob(os.path.join(originals_path, f_name + ".*"))
    if len(f_path_1) > 1:
        Warning("Multiple files containing " + f_name + " found. Using first " +
                "one for EXIF data extraction.")
    elif len(f_path_1) < 1:
        Warning("No file containing " + f_name + " found in " +
                originals_path)
    else:
        print("Extracting EXIF data from " +
              os.path.join(originals_path, f_path_1[0]))
    
    f_path_1 = f_path_1[0]
    f_ext_0 = file.split(".")[-1]
    f_ext_1 = os.path.split(f_path_1)[1].split(".")[-1]
    f_path_2 = os.path.join(output_path, ".".join([f_name , f_ext_1]))
    
    # get EXIF
    image = Image.open(f_path_1)
    exif = image.info["exif"]
    
    print("Savig " + f_name + " as " + f_path_2)
    im = np.asarray(Image.open(f_path_0))
    r = im * (255 / MAX_CLASS_VAL)
    g = 255 - (im * (255 / MAX_CLASS_VAL))
    b = 63.75 + (im * (127.5 / MAX_CLASS_VAL))
    R = Image.fromarray(np.uint8(r))
    G = Image.fromarray(np.uint8(g))
    B = Image.fromarray(np.uint8(b))
    out = Image.merge("RGB", (R, G, B))
    out.save(f_path_2, exif = exif)
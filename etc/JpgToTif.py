# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 23:56:04 2022

@author: Manuel
"""
import os
in_dir = "G:/Masterarbeit/MonitRaw"
out_dir = "G:/Masterarbeit/MonitRawTif"
files = os.listdir(in_dir)

from PIL import Image
for file in files:
    im = Image.open(os.path.join(in_dir, file))
    im.save(os.path.join(out_dir, file.replace("JPG", "tif")), 'TIFF')
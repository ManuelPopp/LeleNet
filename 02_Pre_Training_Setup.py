#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 17 17:56:34 2021

@author: Manuel
Create ImageDataGenerator and define options for training the CNN.
"""
__author__ = "Manuel R. Popp"

os.chdir(wd)

# semantic segmentation
import tensorflow as tf
from tensorflow import keras as ks
tf.__version__
ks.__version__

## make GPU available
phys_devs = tf.config.experimental.list_physical_devices("GPU")
print("N GPUs available: ", len(phys_devs))
if len(phys_devs) >= 1 and False:
    tf.config.experimental.set_memory_growth(phys_devs[0], True)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

## build data generators
N_img = len(list(pathlib.Path(dir_tls(myear = year, dset = "X")) \
                 .glob("**/*.tif")))
N_val = len(list(pathlib.Path(dir_tls(myear = year, dset = "X_val")) \
                 .glob("**/*.tif")))
bs = 5
zeed = 42
args_col = {"data_format" : "channels_last",
            "featurewise_std_normalization" : False,
            "brightness_range" : [0.5, 1.5]
            }
args_aug = {"rotation_range" : 365,
            "width_shift_range" : 0.25,
            "height_shift_range" : 0.25,
            "horizontal_flip" : True,### set to False when training with images that have a top and bottom!
            "vertical_flip" : True,
            "featurewise_center" : False
            }
args_aug["fill_mode"] =  "constant" if no_data_class == True else "reflect"

args_flow = {"class_mode" : None,
             "batch_size" : bs,
             "target_size" : (imgr, imgc),
             "seed" : zeed
             }
### training data generator
X_generator = ks.preprocessing.image.ImageDataGenerator(rescale = 1.0/255.0,
                                                        **args_aug,
                                                        **args_col)
#X_generator.fit(dir_tls(myear = year, dset = "X"))
X_gen = X_generator.flow_from_directory(directory = os.path.dirname(
                        dir_tls(myear = year, dset = "X")),
                        color_mode = "rgb",
                        **args_flow)

if no_data_class == True: args_aug["cval"] = NoDataValue
y_generator = ks.preprocessing.image.ImageDataGenerator(**args_aug,
                                                        dtype = "uint8")
y_gen = y_generator.flow_from_directory(directory = os.path.dirname(
                        dir_tls(myear = year, dset = "y")),
                        color_mode = "grayscale",
                        interpolation = "nearest",
                        **args_flow)

train_generator = zip(X_gen, y_gen)

### validation data generator
X_val_generator = ks.preprocessing.image.ImageDataGenerator(rescale = 1.0/255.0)
X_val_gen = X_generator.flow_from_directory(directory = os.path.dirname(
                        dir_tls(myear = year, dset = "X_val")),
                        color_mode = "rgb",
                        **args_flow)

y_val_generator = ks.preprocessing.image.ImageDataGenerator(dtype = "uint8")
y_val_gen = y_generator.flow_from_directory(directory = os.path.dirname(
                        dir_tls(myear = year, dset = "y_val")),
                        color_mode = "grayscale",
                        interpolation = "nearest",
                        **args_flow)

val_generator = zip(X_val_gen, y_val_gen)

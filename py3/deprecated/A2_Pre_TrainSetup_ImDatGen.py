#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 17 17:56:34 2021

@author: Manuel
Create ImageDataGenerator and define options for training the CNN.
"""
__author__ = "Manuel R. Popp"

os.chdir(wd)

# import modules---------------------------------------------------------------
import tensorflow as tf
from tensorflow import keras as ks
tf.__version__
ks.__version__

## make GPU available----------------------------------------------------------
phys_devs = tf.config.experimental.list_physical_devices("GPU")
print("N GPUs available: ", len(phys_devs))
if len(phys_devs) >= 1 and False:
    tf.config.experimental.set_memory_growth(phys_devs[0], True)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

## general options/info--------------------------------------------------------
N_img = len(list(pathlib.Path(dir_tls(myear = year, dset = "X")) \
                 .glob("**/*.tif")))
N_val = len(list(pathlib.Path(dir_tls(myear = year, dset = "X_val")) \
                 .glob("**/*.tif")))
bs = 5
zeed = 42

## build image data generators-------------------------------------------------
args_col = {"data_format" : "channels_last",
            "featurewise_std_normalization" : False,
            "brightness_range" : [0.75, 1.25]
            }
args_aug = {#"rotation_range" : 365,
            #"width_shift_range" : 0.25,
            #"height_shift_range" : 0.25,
            "horizontal_flip" : True,#-> False when images have top and bottom!
            "vertical_flip" : True,
            "featurewise_center" : False
            }
args_aug["fill_mode"] =  "constant" if no_data_class == True else "reflect"

args_flow = {"class_mode" : None,
             "batch_size" : bs,
             "target_size" : (imgr, imgc),
             "seed" : zeed
             }

### training data generator----------------------------------------------------
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
                                                        dtype = "uint8"
                                                        )
y_gen = y_generator.flow_from_directory(directory = os.path.dirname(
                        dir_tls(myear = year, dset = "y")),
                        color_mode = "grayscale",
                        interpolation = "nearest",
                        **args_flow)

train_generator = zip(X_gen, y_gen)

### validation data generator--------------------------------------------------
X_val_generator = ks.preprocessing.image.ImageDataGenerator(rescale = 1.0/255.0
                                                            )
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

# define weighted categorical crossentropy loss function-----------------------
from PIL import Image
def calculate_weights(directory, n_classes):
    imgs = list(pathlib.Path(directory).glob("**/*.tif"))
    weights = np.array([0] * n_classes)
    gravity = 1
    for img in imgs:
        im = Image.open(img)
        vals = np.array(im.getdata(), dtype = np.uint8)
        unique, counts = np.unique(vals, return_counts = True)
        classweights = np.array([0] * n_classes)
        classweights[unique.astype(int) - 1] = counts
        weights = ((weights * gravity) + classweights) / (gravity + 1)
        gravity += 1
    return weights

def estimate_weights(directory, n_classes, N = 500):
    import random
    imgs = list(pathlib.Path(directory).glob("**/*.tif"))
    weights = np.array([0] * n_classes)
    gravity = 1
    for i in range(N):
        x = random.randint(0, (len(imgs) - 1))
        im = Image.open(imgs[x])
        vals = np.array(im.getdata(), dtype = np.uint8)
        unique, counts = np.unique(vals, return_counts = True)
        classweights = np.array([0] * n_classes)
        classweights[unique.astype(int) - 1] = counts
        weights = ((weights * gravity) + classweights) / (gravity + 1)
        gravity += 1
    return weights
    
WEIGHTS = calculate_weights(os.path.dirname(dir_tls(myear = year, dset = "y")),
                     N_CLASSES)
### inverse frequency as weights
inv_weights = tf.constant((1 / (WEIGHTS + 0.01)), dtype = tf.float32,
                          shape = [1, 1, 1, N_CLASSES])

def wcc_loss(y_true, y_pred, n_classes = N_CLASSES, w = inv_weights):
    # one-hot-encode mask
    onehot = tf.one_hot(indices = tf.cast(y_true, dtype = tf.uint8),
                        depth = n_classes)
    y_pred = tf.cast(y_pred, dtype = tf.float32)
    # divide by sum over last axis to make class probabilities sum up to 1
    y_pred = y_pred / tf.keras.backend.sum(y_pred, axis = -1, keepdims = True)
    # clip to prevent NaN and Inf
    y_pred = tf.keras.backend.clip(y_pred, tf.keras.backend.epsilon(),
                                   1 - tf.keras.backend.epsilon())
    # scale weights so they sum up to 1
    w = w / tf.keras.backend.sum(y_pred, axis = -1, keepdims = True)
    # calculate loss
    wloss = tf.squeeze(onehot) * tf.keras.backend.log(y_pred) * w
    #cc_loss = ks.losses.CategoricalCrossentropy(tf.squeeze(onehot), y_pred)
    wloss = -tf.keras.backend.sum(wloss, -1)
    return wloss#tf.reduce_mean(wloss)

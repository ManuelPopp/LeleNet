#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 17 17:56:34 2021

@author: Manuel
Full implementation
runfrom terminal: python ~/LeleNet/py3/LeleNet_trn.py "U-Net" 10 40
"""
__author__ = "Manuel R. Popp"

#### parse arguments-----------------------------------------------------------
# Standard settings for testing
if False:
    xf = "png"
    yf = "png"
    imgr = 256
    imgc = 256
    imgdim = 3
    wd = "home"
    mdl = "UNet"
    bs = 1
    epochz = 3
    year = "03_2021"
    optmer = "rms"

# Import arguments
import argparse

def parseArguments():
    parser = argparse.ArgumentParser()
    # Positional mandatory arguments
    parser.add_argument("model", help = "Model; one in U-Net, FCDenseNet)",\
                        type = str)
    parser.add_argument("bs", help = "Batchsize (int)",\
                        type = int)
    parser.add_argument("ep", help = "Training epochs (int)",\
                        type = int)
    # Optional arguments
    parser.add_argument("-lr", "--lr",\
                        help = "Initial learning rate (float).",\
                            type = float, default = 1e-3)
    parser.add_argument("-esp", "--esp",\
                        help = "Early stopping patience (int).",\
                            type = int, default = None)
    parser.add_argument("-op", "--op",\
                        help = "Optimizer. 'Adam', 'rms', or 'sgd'.",\
                            type = str, default = "rms")
    parser.add_argument("-xf", "--xf",\
                        help = "Image format; either png, jpg, or tif.",\
                            type = str, default = "png")
    parser.add_argument("-yf", "--yf",\
                        help = "Image format; either png, jpg, or tif.",\
                            type = str, default = "png")
    parser.add_argument("-imgr", "--imgr",\
                        help = "Image x resolution (rows).", type = int,\
                            default = None)
    parser.add_argument("-imgc", "--imgc",\
                        help = "Image y resolution (columns).", type = int,\
                            default = None)
    parser.add_argument("-imgd", "--imgd",\
                        help = "Image dimensions (rows = columns).",\
                            type = int, default = None)
    parser.add_argument("-imgdim", "--imgdim",\
                        help = "X image dimensions (colours).", type = int,\
                            default = 3)
    parser.add_argument("-ww", "--ww",\
                        help = ("Weights scaling factor. Inverse weights =" +\
                            "1/(weights**ww) or 1/math.log(weights, ww)"), \
                            type = float, default = 0.5)
    parser.add_argument("-ws", "--ws",\
                        help = ("Weight scaling (either 'exp' or 'log'."), \
                            type = str, default = "exp")
    parser.add_argument("-wd", "--wd",\
                        help = "Alternative working directory.", type = str,\
                            default = "")
    parser.add_argument("-yr", "--yr",\
                        help = ("Sampling date of the data" +\
                                "as MM_YYYY. Default: '03_2021'"),\
                            type = str, default = "03_2021")
    parser.add_argument("-r", "--r",\
                        help = ("Resume from checkpoint. Either 'f' (False;" +\
                                " default), 't' (True), or date of a specif" +\
                                    "ic training event (folder name)."),\
                            type = str, default = "f")
    # Parse arguments
    args = parser.parse_args()
    return args
if __name__ == "__main__":
    # Parse the arguments
    args = parseArguments()

mdl = args.model
bs = args.bs
epochz = args.ep
init_lr = args.lr
es_patience = args.esp if args.esp is not None else epochz
optmer = args.op
xf = args.xf
yf = args.yf
imgdim = args.imgdim
ww = args.ww
ws = args.ws
wd = args.wd
year = args.yr
resume_training = args.r

# case insensitive arguments
mdl, optmer, xf, yf, wd, resume_training = mdl.casefold(), optmer.casefold(),\
    xf.casefold(), yf.casefold(), wd.casefold(), resume_training.casefold()

#### basic settings------------------------------------------------------------
import platform, sys, datetime, pathlib, os
OS = platform.system()
OS_version = platform.release()
py_version = sys.version
t_start = datetime.datetime.utcnow()
import tensorflow as tf
print("Running on " + OS + " " + OS_version + ".\nPython version: " +
      py_version + "\nTensorflow version: " + tf.__version__ +
      "\nUTC time (start): " + str(t_start) +
      "\nLocal time (start): " + str(datetime.datetime.now()))

# Model (one of "mod_UNet", "mod_FCD")
if mdl in ["u-net", "unet", "mod_unet", "mod_u-net", "u_net"]:
    mod = "mod_UNet"
elif mdl in ["fcd", "fcdensenet", "fc-densenet", "fc-dense-net"]:
    mod = "mod_FCD"
else:
    raise ValueError("Unexpected input for argument 'model': " + str(mdl))

### general directory functions------------------------------------------------
import numpy as np
if wd == "home":
    if OS == "Linux":
        if platform.release() == "4.18.0-193.60.2.el8_2.x86_64":
            wd = "/home/kit/ifgg/mp3890/LeleNet"
        else:
            wd = "/home/manuel/Nextcloud/Masterarbeit"
    elif OS == "Windows":
        wd = os.path.join("C:\\", "Users", "Manuel",\
                          "Nextcloud", "Masterarbeit")
    else:
        raise Exception("OS not detected.")
elif wd == "":
    pydir = os.path.dirname(os.path.realpath(__file__))
    wd = os.path.dirname(pydir)
else:
    wd = args.wd

def dir_fig(fig_id = None):
    if fig_id == None:
        return os.path.join(wd, "fig")
    else:
        return os.path.join(wd, "fig", fig_id)

def dir_dat(dat_id = None):
    if dat_id == None:
        return os.path.join(wd, "dat")
    else:
        dat_id = dat_id.split(",")
        return os.path.join(wd, "dat", *dat_id)

def dir_out(*out_id):
    if len(out_id) < 1:
        return os.path.join(wd, "out")
    else:
        out_lst = list(out_id)
        out_ids = os.path.sep.join(out_lst)
        return os.path.join(wd, "out", out_ids)

def dir_var(pkl_name = None):
    if pkl_name == None:
        return os.path.join(wd, "py3", "vrs")
    else:
        return os.path.join(wd, "py3", "vrs", pkl_name + ".pkl")

def save_var(variables, name):
    import pickle
    os.makedirs(dir_var(), exist_ok = True)
    with open(dir_var(pkl_name = name), "wb") as f:
        pickle.dump(variables, f)

def get_var(name):
    import pickle
    with open(dir_var(pkl_name = name), "rb") as f:
        return pickle.load(f)
    
os.chdir(wd)

with open(dir_out("System_info.txt"), "w") as f:
    f.write("Most recent run on " + OS + " " + OS_version +
            ".\nPython version: " +
            py_version + "\nTensorflow version: " + tf.__version__ +
            "\nUTC time (start): " + str(t_start) +
            "\nLocal time (start): " + str(datetime.datetime.now()))

#### data preparation directory functions--------------------------------------
def dir_omk(plot_id = None, myear = None, type_ext = ""):
    # returns list!
    if plot_id == None:
        if myear == None:
            return dir_dat("omk")
        else:
            return os.path.join(dir_dat("omk"), myear)
    else:
        if myear == None:
            return list(pathlib.Path(dir_dat("omk")) \
                        .glob("**/*" + plot_id + type_ext + ".tif"))
        else:
            return list(pathlib.Path(os.path.join(dir_dat("omk"), myear)) \
                        .glob("**/*" + plot_id + type_ext + ".tif"))

def dir_tls(myear = None, dset = None, plot_id = None):
    if plot_id == None:
        if myear == None:
            if dset == None:
                return dir_dat("tls")
            else:
                return dir_dat("tls")
                raise Exception("Missing year. Returning tile directory.")
        else:
            if dset == None:
                return os.path.join(dir_dat("tls"), myear)
            else:
                return os.path.join(dir_dat("tls"), myear, dset, "0")
    else:
        if myear == None:
            return dir_dat("tls")
            raise Exception("Missing year. Returning tile directory.")
        else:
            if dset == None:
                return os.path.join(dir_dat("tls"), myear)
                raise Exception("Missing dset (X or y)." +\
                                "Returning tile directory.")
            else:
                return os.path.join(dir_dat("tls"), myear, dset, "0", plot_id)
def toINT(filename):
    imgINT = filename.astype("uint8")
    return imgINT

# get tile dimensions if not specified-----------------------------------------
from PIL import Image
if (args.imgr is None or args.imgc is None) and args.imgd is None:
    imgs = list(pathlib.Path(os.path.dirname(dir_tls(myear = year,\
                                                     dset = "y")))\
                .glob("**/*." + yf))
    im = Image.open(imgs[0])
    w, h = im.size
    im.close()

# image dimensions
if args.imgr != args.imgc:
    print("Warning: Arguments imgr and imgc do not match.")
if args.imgr is not None:
    imgr = args.imgr
else:
    imgr = h
if args.imgc is not None:
    imgc = args.imgc
else:
    imgc = w

if args.imgd is not None:
    print("Argument imgd set. imgd overwrites imgr and imgc.")
    imgr = args.imgd
    imgc = args.imgd

# Data preparation-------------------------------------------------------------
### Run file DataPreparation.py
###  read dictionary to group species to classes, if need be
import pandas as pd
specdict = pd.read_excel(dir_dat("xls,SpeciesList.xlsx"),
                         sheet_name = "Dictionary", header = 0)
group_species = False
# exec(open("A1_DataPreparation.py").read())

## load information generated during data preparation--------------------------
classes, classes_decoded, NoDataValue, no_data_class, abc = get_var("ClassIDs")
N_CLASSES = len(classes) if no_data_class or abc else len(classes) + 1

# Setup for training-----------------------------------------------------------
os.chdir(os.path.join(wd, "py3"))
os.chdir(wd)

# import modules---------------------------------------------------------------
#already done in A0_LeleNet.py: import tensorflow as tf
#import tensorflow_io as tfio
from tensorflow import keras as ks
AUTOTUNE = tf.data.experimental.AUTOTUNE
tf.__version__
ks.__version__

## make GPU available----------------------------------------------------------
phys_devs = tf.config.experimental.list_physical_devices("GPU")
print("N GPUs available: ", len(phys_devs))
#if len(phys_devs) >= 1 and False:
#    tf.config.experimental.set_memory_growth(phys_devs[0], True)
#else:
#    #os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
#    my_devices = tf.config.experimental.list_physical_devices(device_type = "CPU")
#    tf.config.experimental.set_visible_devices(devices = my_devices, device_type = "CPU")
#    print("No GPUs used.")

## general options/info--------------------------------------------------------
N_img = len(list(pathlib.Path(dir_tls(myear = year, dset = "X")) \
                 .glob("**/*." + xf)))
N_val = len(list(pathlib.Path(dir_tls(myear = year, dset = "X_val")) \
                 .glob("**/*." + xf)))
zeed = 42

## build data loader-----------------------------------------------------------
def parse_image(img_path: str) -> dict:
    # read image
    image = tf.io.read_file(img_path)
    #image = tfio.experimental.image.decode_tiff(image)
    if xf == "png":
        image = tf.image.decode_png(image, channels = 3)
    else:
        image = tf.image.decode_jpeg(image, channels = 3)
    image = tf.image.convert_image_dtype(image, tf.uint8)
    #image = image[:, :, :-1]
    # read mask
    mask_path = tf.strings.regex_replace(img_path, "X", "y")
    mask_path = tf.strings.regex_replace(mask_path, "X." + xf, "y." + yf)
    mask = tf.io.read_file(mask_path)
    #mask = tfio.experimental.image.decode_tiff(mask)
    mask = tf.image.decode_png(mask, channels = 1)
    #mask = mask[:, :, :-1]
    mask = tf.where(mask == 255, np.dtype("uint8").type(NoDataValue), mask)
    return {"image": image, "segmentation_mask": mask}

train_dataset = tf.data.Dataset.list_files(
    dir_tls(myear = year, dset = "X") + "/*." + xf, seed = zeed)
train_dataset = train_dataset.map(parse_image)

val_dataset = tf.data.Dataset.list_files(
    dir_tls(myear = year, dset = "X_val") + "/*." + xf, seed = zeed)
val_dataset = val_dataset.map(parse_image)

## data transformations--------------------------------------------------------
@tf.function
def normalise(input_image: tf.Tensor, input_mask: tf.Tensor) -> tuple:
    input_image = tf.cast(input_image, tf.float32) / 255.0
    return input_image, input_mask

@tf.function
def load_image_train(datapoint: dict) -> tuple:
    input_image = tf.image.resize(datapoint["image"], (imgr, imgc))
    input_mask = tf.image.resize(datapoint["segmentation_mask"], (imgr, imgc))
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)
    # more experimental data augmentation
#    if tf.random.uniform(()) > 0.5:
#        input_image = tf.image.flip_up_down(input_image)
#        input_mask = tf.image.flip_up_down(input_mask)
#    input_image = tf.image.random_brightness(input_image, max_delta = 0.2)
#    input_image = tf.image.random_contrast(input_image, lower = 0.0, \
#                                           upper = 0.05)
#    input_image = tf.image.random_saturation(input_image, lower = 0.0, \
#                                           upper = 0.05)
    input_image, input_mask = normalise(input_image, input_mask)
    return input_image, input_mask

@tf.function
def load_image_test(datapoint: dict) -> tuple:
    input_image = tf.image.resize(datapoint["image"], (imgr, imgc))
    input_mask = tf.image.resize(datapoint["segmentation_mask"], (imgr, imgc))
    input_image, input_mask = normalise(input_image, input_mask)
    return input_image, input_mask

## create datasets-------------------------------------------------------------
buff_size = 1000
dataset = {"train": train_dataset, "val": val_dataset}
# -- Train Dataset --#
dataset["train"] = dataset["train"]\
    .map(load_image_train, num_parallel_calls = tf.data.experimental.AUTOTUNE)
dataset["train"] = dataset["train"].shuffle(buffer_size = buff_size,
                                            seed = zeed)
dataset["train"] = dataset["train"].repeat()
dataset["train"] = dataset["train"].batch(bs)
dataset["train"] = dataset["train"].prefetch(buffer_size = AUTOTUNE)
#-- Validation Dataset --#
dataset["val"] = dataset["val"].map(load_image_test)
dataset["val"] = dataset["val"].repeat()
dataset["val"] = dataset["val"].batch(bs)
dataset["val"] = dataset["val"].prefetch(buffer_size = AUTOTUNE)

print(dataset["train"])
print(dataset["val"])

# define weighs for categorical crossentropy loss function---------------------
def calculate_weights(directory, n_classes):
    imgs = list(pathlib.Path(directory).glob("**/*." + yf))
    weights = np.array([0] * n_classes)
    gravity = 0
    for img in imgs:
        im = Image.open(img)
        vals = np.array(im.getdata(), dtype = np.uint8)
        unique, counts = np.unique(vals, return_counts = True)
        classweights = np.array([0] * n_classes)
        classweights[unique.astype(int)] = counts
        weights = ((weights * gravity) + classweights) / (gravity + 1)
        gravity += 1
        im.close()
    return weights

def estimate_weights(directory, n_classes, N = 500):
    import random
    imgs = list(pathlib.Path(directory).glob("**/*." + yf))
    weights = np.array([0] * n_classes)
    gravity = 1
    for i in range(N):
        x = random.randint(0, (len(imgs) - 1))
        im = Image.open(imgs[x])
        vals = np.array(im.getdata(), dtype = np.uint8)
        unique, counts = np.unique(vals, return_counts = True)
        classweights = np.array([0] * n_classes)
        classweights[unique.astype(int)] = counts
        weights = ((weights * gravity) + classweights) / (gravity + 1)
        gravity += 1
        im.close()
    return weights

import glob, time
if os.path.isfile(dir_var("weights")):
    print("Loading class weights...")
    WEIGHTS, weights_timestamp = get_var("weights")
    print("Checking class weights timestamp...")
    latest_mod = max(glob.glob(dir_tls(myear = year, dset = "y") + \
                               os.path.sep + "*"), key = os.path.getctime)
    img_mod_timestamp = os.path.getmtime(latest_mod)
    img_mod_timestamp = datetime.datetime.fromtimestamp(img_mod_timestamp)
    if weights_timestamp < img_mod_timestamp:
        print("Weights out of date. Calculating new class weights...")
        WEIGHTS = calculate_weights(
            os.path.dirname(dir_tls(myear = year, dset = "y")), N_CLASSES)
        weights_timestamp = datetime.datetime.now()
        save_var(variables = [WEIGHTS, weights_timestamp],
                 name = "weights")
else:
    print("Calculating class weights...")
    WEIGHTS = calculate_weights(os.path.dirname( \
        dir_tls(myear = year, dset = "y")), N_CLASSES)
    weights_timestamp = datetime.datetime.now()
    save_var(variables = [WEIGHTS, weights_timestamp],
                 name = "weights")
NORMWEIGHTS = WEIGHTS / max(WEIGHTS)
### inverse frequency as weights
#inv_weights = tf.constant((1 / (WEIGHTS + 0.01)), dtype = tf.float32,
#                          shape = [1, 1, 1, N_CLASSES])
import math
inv_weights = (1 / (NORMWEIGHTS + 0.01)**(ww)) if ws == "exp" else \
    [1 / math.log(nw, ww) for nw in NORMWEIGHTS]
inv_weights = inv_weights / max(inv_weights)
print("Calculated the following weights:", inv_weights)

## add weights-----------------------------------------------------------------
def add_sample_weights(image, segmentation_mask):
    class_weights = tf.constant(inv_weights, dtype = tf.float32)
    class_weights = class_weights/tf.reduce_sum(class_weights)
    sample_weights = tf.gather(class_weights,
                               indices = tf.cast(segmentation_mask, tf.int32))
    return image, segmentation_mask, sample_weights

dataset["train"].map(add_sample_weights).element_spec

##############################################################################
#def wcc_loss(y_true, y_pred, n_classes = N_CLASSES, w = inv_weights):
    # one-hot-encode mask (not needed for sparse cce)
    #onehot = tf.one_hot(indices = tf.cast(y_true, dtype = tf.uint8),
    #                    depth = n_classes)
    #y_pred = tf.cast(y_pred, dtype = tf.float32)
    # divide by sum over last axis to make class probabilities sum up to 1
    #y_pred = y_pred / tf.keras.backend.sum(y_pred, axis = -1, keepdims = True)
    # clip to prevent NaN and Inf
    #y_pred = tf.keras.backend.clip(y_pred, tf.keras.backend.epsilon(),
    #                               1 - tf.keras.backend.epsilon())
    # scale weights so they sum up to 1
    #w = w / tf.keras.backend.sum(y_pred, axis = -1, keepdims = True)
#    y_true = tf.cast(y_true, tf.int32)
#    if len(y_pred.shape) == len(y_true.shape):
#        y_true = tf.squeeze(y_true, [-1])
#    w = tf.cast(w, y_pred.dtype)
    # calculate loss
    #wloss = tf.squeeze(onehot) * tf.keras.backend.log(y_pred) * w
#    scce_loss = ks.losses.SparseCategoricalCrossentropy(y_true, y_pred)
#    wloss = tf.math.divide_no_nan(tf.reduce_sum(scce_loss * w),
#                                  tf.reduce_sum(w))
#    return wloss

# Get model--------------------------------------------------------------------
os.chdir(os.path.join(wd, "py3"))
if mod == "mod_UNet":
    def UNet(n_classes, input_shape = (imgr, imgc, imgdim), dropout = 0.5, \
             filters = 64, \
         ops = {"activation" : "relu",
                "padding" : "same",
                "kernel_initializer" : "he_normal"
        }):
        # input layer
        inputz = ks.layers.Input(shape = input_shape)
        
        # encoder part
        ## 1st convolution
        c1 = ks.layers.Conv2D(filters, (3, 3), **ops)(inputz)
        c1 = ks.layers.Conv2D(filters, (3, 3), **ops)(c1)
        ## 1st max pooling
        p1 = ks.layers.MaxPooling2D(pool_size = (2, 2))(c1)
        
        ## 2nd convolution
        c2 = ks.layers.Conv2D(filters*2, (3, 3), **ops)(p1)
        c2 = ks.layers.Conv2D(filters*2, (3, 3), **ops)(c2)
        ## 2nd max pooling
        p2 = ks.layers.MaxPooling2D(pool_size = (2, 2))(c2)
        
        ## 3rd convolution
        c3 = ks.layers.Conv2D(filters*4, (3, 3), **ops)(p2)
        c3 = ks.layers.Conv2D(filters*4, (3, 3), **ops)(c3)
        ## 3rd max pooling
        p3 = ks.layers.MaxPooling2D(pool_size = (2, 2))(c3)
        
        ## 4th convolution
        c4 = ks.layers.Conv2D(filters*8, (3, 3), **ops)(p3)
        c4 = ks.layers.Conv2D(filters*8, (3, 3), **ops)(c4)
        ## Drop
        d4 = ks.layers.Dropout(dropout)(c4)
        ## 4th max pooling
        p4 = ks.layers.MaxPooling2D(pool_size = (2, 2))(d4)
        
        ## 5th convolution
        c5 = ks.layers.Conv2D(filters*16, (3, 3), **ops)(p4)
        c5 = ks.layers.Conv2D(filters*16, (3, 3), **ops)(c5)
        ## Drop
        d5 = ks.layers.Dropout(dropout)(c5)
        
        # decoder part
        ## 1st up convolution
        us6 = ks.layers.UpSampling2D(size = (2, 2))(d5)
        up6 = ks.layers.Conv2D(filters*8, (2, 2), **ops)(us6)
        ## merge
        ct6 = ks.layers.concatenate([d4, up6], axis = 3)
        uc6 = ks.layers.Conv2D(filters*8, (3, 3), **ops)(ct6)
        uc6 = ks.layers.Conv2D(filters*8, (3, 3), **ops)(uc6)
        
        ## 2nd up convolution
        us7 = ks.layers.UpSampling2D(size = (2, 2))(uc6)
        up7 = ks.layers.Conv2D(filters*4, (2, 2), **ops)(us7)
        ## merge
        ct7 = ks.layers.concatenate([c3, up7], axis = 3)
        uc7 = ks.layers.Conv2D(filters*4, (3, 3), **ops)(ct7)
        uc7 = ks.layers.Conv2D(filters*4, (2, 2), **ops)(uc7)
         
        ## 3rd up convolution
        us8 = ks.layers.UpSampling2D(size = (2, 2))(uc7)
        up8 = ks.layers.Conv2D(filters*2, (2, 2), **ops)(us8)
        ## merge
        ct8 = ks.layers.concatenate([c2, up8], axis = 3)
        uc8 = ks.layers.Conv2D(filters*2, (3, 3), **ops)(ct8)
        uc8 = ks.layers.Conv2D(filters*2, (3, 3), **ops)(uc8)
         
        ## 4th up convolution
        us9 = ks.layers.UpSampling2D(size = (2, 2))(uc8)
        up9 = ks.layers.Conv2D(filters, (2, 2), **ops)(us9)
        ## merge
        ct9 = ks.layers.concatenate([c1, up9], axis = 3)
        uc9 = ks.layers.Conv2D(filters, (3, 3), **ops)(ct9)
        uc9 = ks.layers.Conv2D(filters, (3, 3), **ops)(uc9)
        uc9 = ks.layers.Conv2D(2, (3, 3), **ops)(uc9)
        
        # output layer
        if n_classes > 2:
            n_out =  n_classes
            activ = "softmax"
        else:
            n_out = 1
            activ = "sigmoid"
        outputz = ks.layers.Conv2D(n_out, 1, activation = activ)(uc9)
    
        model = ks.Model(inputs = [inputz], outputs = [outputz])
        print(model.summary())
        print(f'Total number of layers: {len(model.layers)}')
        return model

    # get model
    model = UNet(n_classes = N_CLASSES)
    
    # directory to save model
    os.makedirs(dir_out("mod_UNet"), exist_ok = True)
elif mod == "mod_FCD":
    def BN_ReLU_Conv(inputs, n_filters, filter_size = 3, dropout_p = 0.2):
        l = ks.layers.BatchNormalization()(inputs)
        l = ks.layers.Activation("relu")(l)
        l = ks.layers.Conv2D(n_filters, filter_size, activation = None,
                             padding = "same", 
                             kernel_initializer = 'he_uniform') (l)
        if dropout_p != 0.0:
            l = ks.layers.Dropout(dropout_p)(l)
        return l
    
    def TransitionDown(inputs, n_filters, dropout_p = 0.2):
        l = BN_ReLU_Conv(inputs, n_filters, filter_size = 1,\
                         dropout_p = dropout_p)
        l = ks.layers.MaxPool2D(pool_size = (2, 2))(l)
        return l
    
    def TransitionUp(skip_connection, block_to_upsample, n_filters_keep):
        l = ks.layers.concatenate(block_to_upsample)
        l = ks.layers.Conv2DTranspose(n_filters_keep, kernel_size = (3, 3),
                                      strides = (2, 2), padding = "same", 
                                      kernel_initializer = "he_uniform")(l)
        l = ks.layers.concatenate([l, skip_connection])
        return l
    
    def FCDense(n_classes, input_shape = (imgr, imgc, imgdim),
                n_filters_first_conv = 48, n_pool = 4, growth_rate = 12,
                n_layers_per_block = 5, dropout_p = 0.2):
        """
        Original note from the authors of the FC-DenseNet:
        The network consist of a downsampling path, where dense blocks and
        transition down are applied, followed
        by an upsampling path where transition up and dense blocks are applied.
        Skip connections are used between the downsampling path and the
        upsampling path
        Each layer is a composite function of BN - ReLU - Conv and the last
        layer is a softmax layer.
        :param input_shape: shape of the input batch. Only the first dimension
            (n_channels) is needed
        :param n_classes: number of classes
        :param n_filters_first_conv: number of filters for the first
            convolution applied
        :param n_pool: number of pooling layers = number of transition down =
            number of transition up
        :param growth_rate: number of new feature maps created by each layer
            in a dense block
        :param n_layers_per_block: number of layers per block. Can be an int
            or a list of size 2 * n_pool + 1
        :param dropout_p: dropout rate applied after each convolution
            (0. for not using)
        """
        # check n_layers_per_block setting
        if type(n_layers_per_block) == list:
            assert(len(n_layers_per_block) == 2*n_pool + 1)
        elif type(n_layers_per_block) == int:
            n_layers_per_block = [n_layers_per_block]*(2*n_pool + 1)
        else:
            raise ValueError
        # Input layer, m = 3
        inputz = tf.keras.layers.Input(shape = input_shape)
        
        # first convolution; store feature maps in the Tiramisu
        # 3 x 3 convolution, m = 48
        Tiramisu = ks.layers.Conv2D(filters = n_filters_first_conv,
                                    kernel_size = (3, 3), strides = (1, 1),
                                    padding = "same", dilation_rate = (1, 1),
                                    activation = "relu",
                                    kernel_initializer = "he_uniform"
                                    )(inputz)
        n_filters = n_filters_first_conv
        
        # downsampling path, n*(dense block + transition down)
        skip_connection_list = []
        
        for i in range(n_pool):
            ## dense block
            for j in range(n_layers_per_block[i]):
                ### Compute new feature maps
                l = BN_ReLU_Conv(Tiramisu, growth_rate, dropout_p=dropout_p)
                ### and stack it---the Tiramisu is growing
                Tiramisu = ks.layers.concatenate([Tiramisu, l])
                n_filters += growth_rate
            ## store Tiramisu in skip_connections list
            skip_connection_list.append(Tiramisu)
            ## transition Down
            Tiramisu = TransitionDown(Tiramisu, n_filters, dropout_p)
        skip_connection_list = skip_connection_list[::-1]
        
        # bottleneck
        ## store output of subsequent dense block; upsample only these new features
        block_to_upsample = []
        # dense Block
        for j in range(n_layers_per_block[n_pool]):
            l = BN_ReLU_Conv(Tiramisu, growth_rate, dropout_p = dropout_p)
            block_to_upsample.append(l)
            Tiramisu = ks.layers.concatenate([Tiramisu, l])
        
        # upsampling path
        for i in range(n_pool):
            ## Transition Up ( Upsampling + concatenation with the skip connection)
            n_filters_keep = growth_rate * n_layers_per_block[n_pool + i]
            Tiramisu = TransitionUp(skip_connection_list[i], block_to_upsample,
                                 n_filters_keep)
            ## dense Block
            block_to_upsample = []
            for j in range(n_layers_per_block[n_pool + i + 1]):
                l = BN_ReLU_Conv(Tiramisu, growth_rate, dropout_p = dropout_p)
                block_to_upsample.append(l)
                Tiramisu = ks.layers.concatenate([Tiramisu, l])
        
        # output layer; 1x1 convolution, m = number of classes
        if n_classes > 2:
            n_out = n_classes
            activ = "softmax"
        else:
            n_out = 1
            activ = "sigmoid"
        outputz = ks.layers.Conv2D(n_out, 1, \
                                   activation = activ)(Tiramisu)
        
        model = tf.keras.Model(inputs = [inputz], outputs = [outputz])
        print(model.summary())
        print(f'Total number of layers: {len(model.layers)}')
        return model
    
    # get model
    model = FCDense(n_classes = N_CLASSES)
    
    # directory to save model
    os.makedirs(dir_out("mod_FCD"), exist_ok = True)

### logs and callbacks---------------------------------------------------------
# define callbacks
from tensorflow.keras.callbacks import LearningRateScheduler
'''
Simple custom LR decay which would only require the epoch index as an argument:
'''
def step_decay_schedule(initial_lr = 1e-3,
                        decay_factor = 0.75, step_size = 10):
    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch/step_size))
    
    return LearningRateScheduler(schedule)
lr_sched = step_decay_schedule(initial_lr = init_lr,
                               decay_factor = 0.995, step_size = 2)
'''
Using some simple built-in learning rate decay:
'''
lr_sched = ks.optimizers.schedules.ExponentialDecay(
    initial_learning_rate = init_lr,
    # decay after n steps
    decay_steps = np.floor(N_img/bs),
    decay_rate = 0.95)

if optmer == "adam":
    optimizer = ks.optimizers.Adam(learning_rate = lr_sched, clipnorm = 1)
elif optmer == "sgd":
    optimizer = ks.optimizers.SGD(learning_rate = init_lr, clipnorm = 1)
elif optmer == "rms":
    optimizer = ks.optimizers.RMSprop(learning_rate = lr_sched, clipnorm = 1)# FC-DenseNet Optim.
else:
    print("Invalid argument for op: " + optmer + \
          ". Use 'Adam', 'rms', or 'sgd'.")

# list callbacks
now = datetime.datetime.now()
logdir = os.path.join(dir_out("logs"), now.strftime("%y-%m-%d-%H-%M-%S"))
cptdir = os.path.join(dir_out("cpts"), now.strftime("%y-%m-%d-%H-%M-%S"))

cllbs = [
    #ks.callbacks.ReduceLROnPlateau(monitor = "val_loss", factor = 0.2,
    #                               patience = 5, min_lr = 0.001),
    ks.callbacks.EarlyStopping(patience = es_patience),
    ks.callbacks.ModelCheckpoint(os.path.join(cptdir, \
                                              "Epoch.{epoch:02d}.hdf5"),
                                 save_best_only = True),
    ks.callbacks.TensorBoard(log_dir = logdir, histogram_freq = 5)
    ]

# compile model----------------------------------------------------------------
## loss functions
### define dice coefficient
### https://github.com/tensorlayer/tensorlayer/blob/master/tensorlayer/cost.py#L216
def dice_coe(output, target, loss_type = "jaccard",
             axis = (1, 2, 3), smooth = 1):# orig. val. smooth = 1e-5
    inse = tf.reduce_sum(output * target, axis = axis)
    if loss_type == "jaccard":
        l = tf.reduce_sum(output * output, axis = axis)
        r = tf.reduce_sum(target * target, axis = axis)
    elif loss_type == "sorensen":
        l = tf.reduce_sum(output, axis = axis)
        r = tf.reduce_sum(target, axis = axis)
    else:
        raise Exception("Unknow loss_type: " + loss_type)
    dice = (2. * inse + smooth) / (l + r + smooth)
    dice = tf.reduce_mean(dice)
    return 1 - dice
lozz = dice_coe

### get sparse categorical cross entropy
lozz = ks.losses.SparseCategoricalCrossentropy() if N_CLASSES > 2 else\
    ks.losses.BinaryCrossentropy()

## metrics
### get intersect. over union (original function gives error -> updated accor-
### ding to https://stackoverflow.com/a/61826074/11611246)
class UpdatedMeanIoU(tf.keras.metrics.MeanIoU):
    def __init__(self,
                 y_true = None,
                 y_pred = None,
                 num_classes = None,
                 name = None,
                 dtype = None):
        super(UpdatedMeanIoU, self).__init__(num_classes = num_classes,
                                             name = name, dtype = dtype)

    def update_state(self, y_true, y_pred, sample_weight = None):
        y_pred = tf.math.argmax(y_pred, axis = -1)
        return super().update_state(y_true, y_pred, sample_weight)
mIoU = UpdatedMeanIoU(num_classes = N_CLASSES)
#mIoU = ks.metrics.MeanIoU(num_classes = N_CLASSES)

#lozz = wcc_loss
#run_opts = tf.compat.v1.RunOptions(report_tensor_allocations_upon_oom = True)

# resume training or compile new model-----------------------------------------
if resume_training == "f":
    os.makedirs(logdir, exist_ok = True)
    os.makedirs(cptdir, exist_ok = True)
    os.chdir(logdir)
    model.compile(optimizer = optimizer, loss = lozz,
                  metrics = [mIoU#, "sparse_categorical_accuracy"
                             ])#, options = run_opts)
    model.summary()
elif resume_training == "t":
    cpt_folders = [f for f in os.listdir(dir_out("cpts")) \
                   if not f.startswith(".")]
    cpt_dates = [datetime.datetime.strptime(d, "%y-%m-%d-%H-%M-%S"\
                                            ) for d in cpt_folders]
    cpt_folder = max(cpt_dates).strftime("%y-%m-%d-%H-%M-%S")
else:
    cpt_folder = resume_training

if resume_training != "f":
    list_of_files = glob.glob(dir_out("cpts", cpt_folder) + os.path.sep + \
                              "*" + ".hdf5")
    checkpoint = max(list_of_files, key = os.path.getctime)
    try:
        model = ks.models.load_model(checkpoint, \
                                     custom_objects = {"UpdatedMeanIoU": mIoU})
    except:
        print("Failed to load model from", checkpoint)
    all_logs = [dir_out("logs", p) for p in os.listdir(dir_out("logs"))]
    logdir =  max(all_logs, key = os.path.getctime)
    os.chdir(logdir)
    model.compile(optimizer = optimizer, loss = lozz,
                  metrics = [mIoU#, "sparse_categorical_accuracy"
                             ])

# fit model--------------------------------------------------------------------
args_fit = {"epochs" : epochz,
            "steps_per_epoch" : np.ceil(N_img/bs),
            "validation_steps" : np.ceil(N_val/bs),
            "callbacks" : cllbs}
if resume_training != "f":
    try:
        s = checkpoint.find("Epoch.") + len("Epoch.")
        e = checkpoint.find("Epoch.") + len("Epoch.") + 2
        args_fit["initial_epoch"] = int(checkpoint[s : e])
    except:
        print("Error when trying to retreive the epoch number from filename", \
              "'" + checkpoint + "': Unable to find integer at position", \
                  str(checkpoint.find("Epoch.") + len("Epoch.")), "to", \
                      str(len(checkpoint)-5))
if "train_generator" in locals() or "train_generator" in globals():
    model.fit(train_generator,
                     validation_data = val_generator,
                     **args_fit)
else:     
    model.fit(dataset["train"].map(add_sample_weights),
                     validation_data = dataset["val"],
                     **args_fit)

os.chdir(dir_out())

# save model-------------------------------------------------------------------
os.makedirs(dir_out(mod), exist_ok = True)
model.save(dir_out(mod), save_format = "tf", save_traces = True)
print("Model saved to disc.")

#trained_model = ks.models.load_model(dir_out(mod),\
#                                     custom_objects = {"UpdatedMeanIoU": mIoU})

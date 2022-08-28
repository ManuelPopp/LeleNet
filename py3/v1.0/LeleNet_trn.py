#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 17:56:34 2022

@author: Manuel
Full implementation
run in terminal: python3 ~/LeleNet/py3/LeleNet_trn.py "U-Net" 10 40
"""
__author__ = "Manuel R. Popp"

#### parse arguments-----------------------------------------------------------
# Import arguments
import argparse, pickle

def parseArguments():
    parser = argparse.ArgumentParser()
    # Positional mandatory arguments
    parser.add_argument("model", help = "Model; one in U-Net," + \
                        "FCDenseNet, DeepLabV3.", type = str)
    parser.add_argument("bs", help = "Batchsize.",\
                        type = int)
    parser.add_argument("ep", help = "Training epochs.",\
                        type = int)
    # Optional arguments
    parser.add_argument("-lr", "--lr",\
                        help = "Initial learning rate.",\
                            type = float, default = 1e-4)
    parser.add_argument("-lrd", "--lrd",\
                        help = "Learning rate decay factor.",\
                            type = float, default = 0.995)
    parser.add_argument("-lrs", "--lrs",\
                        help = "Learning rate decay step size.",\
                            type = int, default = 2)
    parser.add_argument("-cbm", "--cbm",\
                        help = "Callback metric (used to monitor proress)." + \
                            " Options: None (use metric set in -mx), " + \
                                "'val_loss', or 'val_accuracy'.",\
                            type = str, default = None)
    parser.add_argument("-esp", "--esp",\
                        help = "Early stopping patience.",\
                            type = int, default = None)
    parser.add_argument("-op", "--op",\
                        help = "Optimizer. 'Adam', 'rms', or 'sgd'.",\
                            type = str, default = "rms")
    parser.add_argument("-ki", "--ki",\
                        help = "Kernel initialiser.",\
                            type = str, default = None)
    parser.add_argument("-dr", "--dr",\
                        help = "Dropout rate.",\
                            type = float, default = 0.2)
    parser.add_argument("-FCD", "--FCD",\
                        help = "Type of FC-DenseNet (settings from the " + \
                            "original publication).", type = int, default = 0)
    parser.add_argument("-FCD_layers", "--FCD_layers",\
                        help = "Layers per block (FC-DenseNet).",\
                            nargs = "+", type = int, default = 5)
    parser.add_argument("-FCD_growth_rate", "--FCD_growth_rate",\
                        help = "Growth rate (FC-DenseNet).",\
                            type = int, default = 12)
    parser.add_argument("-FCD_filters", "--FCD_filters",\
                        help = "N filters first convolution (FC-DenseNet).",\
                            type = int, default = 48)
    parser.add_argument("-FCD_n_pool", "--FCD_n_pool",\
                        help = "n_pool (FC-DenseNet).",\
                            type = int, default = 4)
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
                        help = "X colour bands (channels).", type = int,\
                            default = 3)
    parser.add_argument("-nb", "--nb",\
                        help = "Number of bands (channels).", type = int,\
                            default = 3)
    parser.add_argument("-nc", "--nc",\
                        help = "Number of classes.", type = int,\
                            default = None)
    parser.add_argument("-lc", "--lc",\
                        help = "Lowest class value.", type = int,\
                            default = 0)
    parser.add_argument("-ww", "--ww",\
                        help = ("Weights scaling factor. Inverse weights =" +\
                            "1/(weights**ww) or 1/math.log(weights, ww)"), \
                            type = float, default = 0.0)
    parser.add_argument("-ws", "--ws",\
                        help = ("Weight scaling (either 'exp' or 'log'."), \
                            type = str, default = "exp")
    parser.add_argument("-mx", "--mx",\
                        help = ("Metric to track; either mIoU or f1."), \
                            type = str, default = "mIoU")
    parser.add_argument("-wd", "--wd",\
                        help = "Alternative working directory.", type = str,\
                            default = "")
    parser.add_argument("-dd", "--dd",\
                        help = "Read data from alternative parent directory.",\
                            type = str, default = None)
    parser.add_argument("-od", "--od",\
                        help = "Alternative output directory parent path.",\
                            type = str, default = None)
    parser.add_argument("-out", "--out",\
                        help = "Alternative output folder name.", type = str,\
                            default = None)
    parser.add_argument("-yr", "--yr",\
                        help = ("Sampling date of the data" +\
                                "as MM_YYYY. Default: '03_2021'"),\
                            type = str, default = "03_2021")
    parser.add_argument("-r", "--r",\
                        help = ("Resume from checkpoint. Either 'f' (False;" +\
                                " default), 't' (True), or folder name."), \
                            type = str, default = "f")
    parser.add_argument("-save_settings", "--sv",\
                        help = "Save training settings.", \
                            action = "store_false")
    parser.add_argument("-debug", "--debug",\
                        help = "Write successful steps to txt.", \
                            action = "store_true")
    parser.add_argument("-tb", "--tb",\
                        help = "Export graphs to TensorBoard dev.", \
                            action = "store_false")
    parser.add_argument("-tbn", "--tbn",\
                        help = "Alternative TensorBoard experiment name.", \
                            type = str, default = None)
    # Parse arguments
    args = parser.parse_args()
    return args
if __name__ == "__main__":
    # Parse the arguments
    args = parseArguments()

# debug mode
if False:
    saved_args = \
    "C:\\Users\\Manuel\\Nextcloud\\Masterarbeit\\py3\\vrs\\train_settings.pkl"
    with open(saved_args, "rb") as f:
        args = pickle.load(f)
    args.wd = "home"

mdl = args.model
bs = args.bs
epochz = args.ep
init_lr = args.lr
decay_lr = args.lrd
step_lr = args.lrs
cb_metric = args.cbm
monitor_mode = "min" if cb_metric == "val_loss" else "max"
es_patience = args.esp if args.esp is not None else epochz
optmer = args.op
kernel_init = args.ki
drop = args.dr
FCD_type = args.FCD
if type(args.FCD_layers) == int:
    FCD_l = args.FCD_layers
else:
    FCD_l = args.FCD_layers if len(args.FCD_layers) > 1 else args.FCD_layers[0]
FCD_gr = args.FCD_growth_rate
FCD_f = args.FCD_filters
FCD_p = args.FCD_n_pool
xf = args.xf
yf = args.yf
imgdim = args.imgdim
lc = args.lc
ww = args.ww
ws = args.ws
mk = args.mx
wd = args.wd
dd = args.dd
od = args.od
year = args.yr
alternative_out_folder = args.out
resume_training = args.r
tb = args.tb
tbn = args.tbn

# case insensitive arguments
mdl, optmer, xf, yf, wd, mk = mdl.casefold(), optmer.casefold(),\
    xf.casefold(), yf.casefold(), wd.casefold(), mk.casefold()
if resume_training in ["T", "F"]:
    resume_training = resume_training.casefold()

print("Script executed with the following input parameters:", \
    "Dataset:", year, \
    "Model:", mdl, "Batch size:", str(bs), "Epochs:", str(epochz), \
    "Initial learning rate:", str(init_lr), "Learning rate decay:", \
    str(decay_lr),
          "Weights:", str(ww), "Metric:", str(mk))

#### basic settings------------------------------------------------------------
import platform, sys, datetime, pathlib, os
OS = platform.system()
OS_version = platform.release()
py_version = sys.version
t_start = datetime.datetime.utcnow()

import tensorflow as tf
import tensorflow_addons as tfa
print("Running on " + OS + " " + OS_version + ".\nPython version: " +
      py_version + "\nTensorflow version: " + tf.__version__ +
      "\nUTC time (start): " + str(t_start) +
      "\nLocal time (start): " + str(datetime.datetime.now()))

# Model (one of "mod_UNet", "mod_FCD", "mod_DL3")
if mdl in ["u-net", "u_net", "unet", "mod_unet", "mod_u-net"]:
    mod = "mod_UNet"
elif mdl in ["fcd", "fcdensenet", "fc-densenet", "fc-dense-net", "mod_FCD"]:
    mod = "mod_FCD"
elif mdl in ["deep", "deeplab", "deeplabv3+", "deeplabv3plus", "deeplabv3", \
             "dl3", "dlv3", "dlv3+", "mod_DL3"]:
    mod = "mod_DL3"
else:
    raise ValueError("Unexpected input for argument 'model': " + str(mdl))

# FC-DenseNet settings
if FCD_type == 103:
    print("FC-DenseNet103 Jegou et al. 2017")
    FCD_l = [4, 5, 7, 10, 12, 15, 12, 10, 7, 5, 4]
    FCD_gr = 16
    FCD_f = 48
    FCD_p = 5
    #drop = 0.2
elif FCD_type == 67:
    print("FC-DenseNet67 Jegou et al. 2017")
    FCD_l = 5
    FCD_gr = 16
    FCD_f = 48
    FCD_p = 5
elif FCD_type == 56:
    print("FC-DenseNet56 Jegou et al. 2017")
    FCD_l = 4
    FCD_gr = 12
    FCD_f = 48
    FCD_p = 4
elif FCD_type == 2:
    print("FC-DenseNet following Lobo Torres et al. 2020")
    FCD_l = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    FCD_gr = 8
    FCD_f = 32
    FCD_p = 8
    #drop = 0
elif mod == "mod_FCD":
    print("FC-DenseNet custom settings:",
          "\nFCD_l =", str(FCD_l), "\nFCD_gr=", str(FCD_gr), "\nFCD_f =",
          str(FCD_f), "\nFCD_p =", str(FCD_p))
else:
    print("Model: ", mod)

### general directory functions------------------------------------------------
import numpy as np
import re
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
    # assumes this script is placed within a direct subfolder of wd (wd/py3)
    pydir = os.path.dirname(os.path.realpath(__file__))
    wd = os.path.dirname(pydir)
else:
    wd = args.wd

# assign alternative data and output parent directories if options were set
dd = wd if dd is None or dd == "wd" else dd
od = wd if od is None or od == "wd" else od

def dir_fig(*args):
    fig_dir = os.path.join(wd, "fig")
    if len(args) == 0:
        return fig_dir
    else:
        fig_id = "/".join(args)
        fig_id = re.split("[,/+ ]", fig_id)
        return os.path.join(fig_dir, *fig_id)

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

out_folder_name = alternative_out_folder if alternative_out_folder is not \
    None else year + os.path.sep + mod
def dir_out(*args):
    out_dir = os.path.join(od, "out", out_folder_name)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    if len(args) == 0:
        return out_dir
    else:
        out_id = "/".join(args)
        out_id = re.split("[,/+ ]", out_id)
        return os.path.join(out_dir, *out_id)

def dir_var(pkl_name = None):
    var_dir = os.path.join(wd, "py3", "vrs")
    if pkl_name == None:
        return var_dir
    else:
        return os.path.join(var_dir, pkl_name + ".pkl")

def save_var(variables, name):
    os.makedirs(dir_var(), exist_ok = True)
    with open(dir_var(pkl_name = name), "wb") as f:
        pickle.dump(variables, f)

def get_var(name):
    with open(dir_var(pkl_name = name), "rb") as f:
        return pickle.load(f)
    
os.chdir(wd)

with open(dir_out("System_info.txt"), "w") as f:
    f.write("Most recent run on " + OS + " " + OS_version +
            ".\nPython version: " +
            py_version + "\nTensorflow version: " + tf.__version__ +
            "\nUTC time (start): " + str(t_start) +
            "\nLocal time (start): " + str(datetime.datetime.now()))

if args.sv:
    save_var(args, "train_settings")
    print("Saved training settings.")

# debug mode text output-------------------------------------------------------
def debug_cp(line, debug = args.debug):
    if debug:
        with open(dir_out("Debug.txt"), "a") as cp:
            cp.write(line)

#### data preparation directory functions--------------------------------------
def dir_omk(plot_id = None, myear = None, type_ext = ""):
    # returns list!
    omk_dir = dir_dat("omk")
    if plot_id == None:
        if myear == None:
            return omk_dir
        else:
            return os.path.join(omk_dir, myear)
    else:
        if myear == None:
            return list(pathlib.Path(omk_dir) \
                        .glob("**/*" + plot_id + type_ext + ".tif"))
        else:
            return list(pathlib.Path(os.path.join(omk_dir, myear)) \
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

def save_dataset_info(variables, year = year, info = "dset_info"):
    tile_dir = dir_tls(myear = year)
    os.makedirs(tile_dir, exist_ok = True)
    with open(tile_dir + os.path.sep + info + ".pkl", "wb") as f:
        pickle.dump(variables, f)

def get_dataset_info(dataset = year, info = "dset_info"):
    inf_dir = os.path.join(dir_tls(myear = dataset), info + ".pkl")
    with open(inf_dir, "rb") as f:
        return pickle.load(f)

def toINT(filename):
    imgINT = filename.astype("uint8")
    return imgINT

if not os.path.exists(dir_out()):
    os.makedirs(dir_out())

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
'''
import pandas as pd
specdict = pd.read_excel(dir_xls("SpeciesList.xlsx"),
                         sheet_name = "Dictionary", header = 0)
'''
# exec(open("A1_DataPreparation.py").read())

## load information generated during data preparation--------------------------
classes, classes_decoded, NoDataValue, no_data_class, abc = get_dataset_info()
N_CLASSES = len(classes) if no_data_class or abc else len(classes) + 1

if args.nc is not None:
    N_CLASSES = args.nc

# Setup for training-----------------------------------------------------------
os.chdir(wd)

# import modules---------------------------------------------------------------
debug_cp(line = "Import AUTOTUNE...\n")
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
#    my_devices = tf.config.experimental.list_physical_devices( \
#                                                    device_type = "CPU")
#    tf.config.experimental.set_visible_devices(devices = my_devices, \
#                                                device_type = "CPU")
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
    if xf == "png":
        image = tf.image.decode_png(image, channels = imgdim)
    elif xf == "jpg":
        image = tf.image.decode_jpeg(image, channels = imgdim)
    elif xf == "tif":
        import tensorflow_io as tfio
        image = tfio.experimental.image.decode_tiff(image)
    else:
        print("Invalid X data format. Allowed formats: png, jpg, tif")
    # read mask
    mask_path = tf.strings.regex_replace(img_path, "X", "y")
    mask_path = tf.strings.regex_replace(mask_path, "X." + xf, "y." + yf)
    mask_path = tf.strings.regex_replace(mask_path, "image", "mask")
    mask = tf.io.read_file(mask_path)
    if yf == "png":
        mask = tf.image.decode_png(mask, channels = 1)
    elif yf == "tif":
        import tensorflow_io as tfio
        mask = tfio.experimental.image.decode_tiff(mask)
    else:
        print("Invalid y data format. Allowed formats: png, tif")
    mask = tf.where(mask == 255, np.dtype("uint8").type(NoDataValue), mask)
    return {"image": image, "segmentation_mask": mask}

train_dataset = tf.data.Dataset.list_files(
    dir_tls(myear = year, dset = "X") + os.path.sep + "*." + xf, seed=zeed)
train_dataset = train_dataset.map(parse_image)

val_dataset = tf.data.Dataset.list_files(
    dir_tls(myear = year, dset = "X_val") + os.path.sep + "*." + xf, seed=zeed)
val_dataset = val_dataset.map(parse_image)

## data transformations--------------------------------------------------------
if lc == 0:
    @tf.function
    def normalise(input_image: tf.Tensor, input_mask: tf.Tensor) -> tuple:
        input_image = tf.cast(input_image, tf.float32) / 255.0
        #input_image = tf.cast(input_image, tf.float32)
        tf.image.per_image_standardization(input_image)
        input_mask = tf.round(input_mask)
        input_mask = tf.cast(input_mask, tf.uint8)
        return input_image, input_mask
else:
    print("Adjusting classes (lowest value != 0). This slows down training.")
    lwst_cls = tf.convert_to_tensor(lc, dtype = tf.uint8)
    @tf.function
    def normalise(input_image: tf.Tensor, input_mask: tf.Tensor) -> tuple:
        input_image = tf.cast(input_image, tf.float32) / 255.0
        #input_image = tf.cast(input_image, tf.float32)
        tf.image.per_image_standardization(input_image)
        input_mask = tf.round(input_mask)
        input_mask = tf.cast(input_mask, tf.uint8)
        input_mask = tf.subtract(input_mask, lwst_cls)
        return input_image, input_mask

@tf.function
def load_image_train(datapoint: dict) -> tuple:
    input_image = tf.image.resize(datapoint["image"], (imgr, imgc))
    input_mask = tf.image.resize(datapoint["segmentation_mask"], \
                                 (imgr, imgc), method = "nearest")
    # random flip
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)
    '''
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_up_down(input_image)
        input_mask = tf.image.flip_up_down(input_mask)
    '''
    ###########################################################################
    ## experimental augmentation
    # calculate upscaling through rotation
    #angle = np.random.rand(1) * 2.0 * np.pi
    angle = np.random.rand(1) * 0.7
    if angle > 0.35:
        angle += (2 * np.pi - 0.7)
    remainder = angle % (0.5 * np.pi)
    scaled_by_rotation = (np.sin(remainder) + np.cos(remainder)).item()
    
    # pad image by rotation scaling factor through reflect padding
    p_rows = int(((imgr * scaled_by_rotation) - imgr) / 2)
    p_cols = int(((imgc * scaled_by_rotation) - imgc) / 2)
    pad = tf.constant([[p_rows, p_rows], [p_cols, p_cols], [0,0]])
    input_image = tf.pad(input_image, pad, "reflect")
    input_mask = tf.pad(input_mask, pad, "reflect")
    
    # rotate image by angle
    input_image = tfa.image.rotate(input_image, \
                                   angle, \
                                       interpolation = "nearest", \
                                           fill_mode = "reflect")
    input_mask = tfa.image.rotate(input_mask, \
                                  angle, \
                                      interpolation = "nearest", \
                                          fill_mode = "reflect")
    
    # random scaling
    max_scaling_factor = 0.1
    random_scaling = (tf.random.uniform(()) * max_scaling_factor) + \
        (1.0 - 0.5*max_scaling_factor)
    
    # clip to original size (central crop as fraction of scaled image)
    fraction = (1.0 / scaled_by_rotation) * random_scaling
    fraction = 1.0 if fraction > 1.0 or fraction <= 0.0 else fraction
    input_image = tf.image.central_crop(input_image, \
                                        central_fraction = fraction)
    input_mask = tf.image.central_crop(input_mask, \
                                       central_fraction = fraction)
    
    # resize to original size
    input_image = tf.image.resize(input_image, (imgr, imgc), \
                                  method = "lanczos3")
    input_mask = tf.image.resize(input_mask, (imgr, imgc), \
                                  method = "nearest")
    ###########################################################################
    # normalise images
    input_image, input_mask = normalise(input_image, input_mask)
    # colour augmentation
    input_image = tf.image.random_brightness(input_image, max_delta = 0.25)
    input_image = tf.image.random_contrast(input_image, lower = 0.5, \
                                           upper = 2.0)
    input_image = tf.image.random_saturation(input_image, lower = 0.6, \
                                           upper = 1.75)
    input_image = tf.clip_by_value(input_image, clip_value_min = 0.0, \
                                   clip_value_max = 1.0)
    return input_image, input_mask

@tf.function
def load_image_test(datapoint: dict) -> tuple:
    input_image = tf.image.resize(datapoint["image"], (imgr, imgc))
    input_mask = tf.image.resize(datapoint["segmentation_mask"], (imgr, imgc))
    input_image, input_mask = normalise(input_image, input_mask)
    return input_image, input_mask

## create datasets-------------------------------------------------------------
debug_cp(line = "Create datasets...")
buff_size = 1000
dataset = {"train": train_dataset, "val": val_dataset}
# train dataset
dataset["train"] = dataset["train"]\
    .map(load_image_train, num_parallel_calls = tf.data.experimental.AUTOTUNE)
dataset["train"] = dataset["train"].shuffle(buffer_size = buff_size,
                                            seed = zeed)
dataset["train"] = dataset["train"].repeat()
dataset["train"] = dataset["train"].batch(bs)
dataset["train"] = dataset["train"].prefetch(buffer_size = AUTOTUNE)
# validation dataset
dataset["val"] = dataset["val"].map(load_image_test)
dataset["val"] = dataset["val"].repeat()
dataset["val"] = dataset["val"].batch(bs)
dataset["val"] = dataset["val"].prefetch(buffer_size = AUTOTUNE)

print(dataset["train"])
print(dataset["val"])

# define weighs for categorical crossentropy loss function---------------------
if lc == 0:
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
    
    def estimate_weights(directory, n_classes, N = 1000):
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
else:
    def calculate_weights(directory, n_classes):
        imgs = list(pathlib.Path(directory).glob("**/*." + yf))
        weights = np.array([0] * n_classes)
        gravity = 0
        for img in imgs:
            im = Image.open(img)
            vals = np.array(im.getdata(), dtype = np.uint8)
            unique, counts = np.unique(vals, return_counts = True)
            classweights = np.array([0] * n_classes)
            classweights[unique.astype(int) - lc] = counts
            weights = ((weights * gravity) + classweights) / (gravity + 1)
            gravity += 1
            im.close()
        return weights

import glob, time
debug_cp(line = "Calculate weights...")
if ww != 0.0:
    if os.path.isfile(os.path.join(dir_tls(myear = year), "weights.pkl")):
        print("Loading class weights...")
        WEIGHTS, weights_timestamp = get_dataset_info(info = "weights")
        print("Checking class weights timestamp...")
        latest_mod = max(glob.glob(dir_tls(myear = year, dset = "y") + \
                                   os.path.sep + "*"), key = os.path.getctime)
        img_mod_timestamp = os.path.getmtime(latest_mod)
        img_mod_timestamp = datetime.datetime.fromtimestamp(img_mod_timestamp)
        if weights_timestamp < img_mod_timestamp:
            print("Weights outdated. Calculating new class weights...")
            WEIGHTS = calculate_weights(
                os.path.dirname(dir_tls(myear = year, dset = "y")), N_CLASSES)
            weights_timestamp = datetime.datetime.now()
            save_dataset_info(variables = [WEIGHTS, weights_timestamp],
                     info = "weights")
    else:
        print("Calculating class weights...")
        WEIGHTS = calculate_weights(os.path.dirname( \
            dir_tls(myear = year, dset = "y")), N_CLASSES)
        weights_timestamp = datetime.datetime.now()
        save_dataset_info(variables = [WEIGHTS, weights_timestamp],
                     info = "weights")
    NORMWEIGHTS = WEIGHTS / max(WEIGHTS)
    ### inverse frequency as weights
    #inv_weights = tf.constant((1 / (WEIGHTS + 0.01)), dtype = tf.float32,
    #                          shape = [1, 1, 1, N_CLASSES])
    import math
    inv_weights = (1 / ((NORMWEIGHTS + 0.01)**(ww))) if ws == "exp" else \
        [1 / math.log(nw, ww) for nw in NORMWEIGHTS]
    inv_weights = inv_weights / max(inv_weights)
    print("Calculated the following weights:", inv_weights)

    ## add weights-------------------------------------------------------------
    def add_sample_weights(image, segmentation_mask, cw = inv_weights):
        class_weights = tf.constant(cw, dtype = tf.float32)
        class_weights = class_weights/tf.reduce_sum(class_weights)
        sample_weights = tf.gather(class_weights,
                                   indices = tf.cast(segmentation_mask, \
                                                     tf.int32))
        return image, segmentation_mask, sample_weights
    print(dataset["train"].map(add_sample_weights).element_spec)

# Get model--------------------------------------------------------------------
debug_cp(line = "Get model...")
if kernel_init is not None:
    k_initializers = { \
        "he_normal" : "he_normal", \
        "he_uniform" : "he_uniform", \
        "random_uniform" : ks.initializers.RandomUniform(minval = 0.0,\
                                                         maxval = 1.0), \
        "truncated_normal" : ks.initializers.TruncatedNormal(mean = 0.0, \
                                                             stddev = 0.05) \
            }
    initializer = k_initializers[kernel_init.casefold()]
    #-------------------------------------------------------------------------
    # U-Net
if mod == "mod_UNet":
    # Suggested by Ronneberget et al. (2015):
    # Gaussian distribution with sd of sqrt(2/N) where N = number incoming
    # nodes of one neuron -> he_normal
    ops = {"padding" : "same"}
    if kernel_init is not None:
        ops["kernel_initializer"] = initializer
    # define model blocks
    #https://github.com/bnsreenu/python_for_image_processing_APEER/blob/master/tutorial119_multiclass_semantic_segmentation.ipynb
    def conv_block(input, num_filters):
        x = ks.layers.Conv2D(num_filters, 3, **ops)(input)
        x = ks.layers.BatchNormalization()(x)
        x = ks.layers.Activation("relu")(x)
        x = ks.layers.Conv2D(num_filters, 3, **ops)(x)
        x = ks.layers.BatchNormalization()(x)
        x = ks.layers.Activation("relu")(x)
        return x

    def encoder_block(input, num_filters):
        x = conv_block(input, num_filters)
        p = ks.layers.MaxPool2D((2, 2))(x)
        return x, p
    
    def decoder_block(input, skip_features, num_filters):
        x = ks.layers.Conv2DTranspose(num_filters, (2, 2), \
                                      strides = 2, **ops)(input)
        return x
    
    def build_unet(input_shape, n_classes, filters = 64):
        inputz = ks.layers.Input(input_shape)
        s1, p1 = encoder_block(inputz, filters)
        s2, p2 = encoder_block(p1, (filters * 2))
        s3, p3 = encoder_block(p2, (filters * 4))
        s4, p4 = encoder_block(p3, (filters * 8))
        b1 = conv_block(p4, (filters * 16))
        d1 = decoder_block(b1, s4, (filters * 8))
        d2 = decoder_block(d1, s3, (filters * 4))
        d3 = decoder_block(d2, s2, (filters * 2))
        d4 = decoder_block(d3, s1, filters)
        activation = "sigmoid" if n_classes < 2 else "softmax"
        outputs = ks.layers.Conv2D(n_classes, 1, padding = "same", \
                                   activation = activation)(d4)
        model = ks.models.Model(inputz, outputs, name = "UNet")
        return model
    model = build_unet((imgr, imgc, imgdim), n_classes = N_CLASSES)
    print(model.summary())
    print(f'Total number of ks.layers: {len(model.layers)}')
    #-------------------------------------------------------------------------
    # FCDenseNet
    #https://github.com/danifranco/EM_Image_Segmentation/blob/master/models/tiramisu.py
elif mod == "mod_FCD":
    if kernel_init is None:
        # following Jegou et al (2017)
        initializer = ks.initializers.HeUniform() # or "he_uniform"
    else:
        initializer = kernel_init
    def BN_ReLU_Conv(inputs, n_filters, filter_size = 3, dropout_p = drop):
        l = ks.layers.BatchNormalization()(inputs)
        l = ks.layers.Activation("relu")(l)
        l = ks.layers.Conv2D(n_filters, filter_size, activation = None,
                             padding = "same", 
                             kernel_initializer = initializer) (l)
        if dropout_p != 0.0:
            l = ks.layers.Dropout(dropout_p)(l)
        return l
    
    def TransitionDown(inputs, n_filters, dropout_p = drop):
        l = BN_ReLU_Conv(inputs, n_filters, filter_size = 1,\
                         dropout_p = dropout_p)
        l = ks.layers.MaxPool2D(pool_size = (2, 2))(l)
        return l
    
    def TransitionUp(skip_connection, block_to_upsample, n_filters_keep):
        l = ks.layers.concatenate(block_to_upsample)
        l = ks.layers.Conv2DTranspose(n_filters_keep, kernel_size = (3, 3),
                                      strides = (2, 2), padding = "same", 
                                      kernel_initializer = initializer)(l)
        l = ks.layers.concatenate([l, skip_connection])
        return l
    
    def FCDense(n_classes, input_shape = (imgr, imgc, imgdim),
                n_filters_first_conv = FCD_f, n_pool = FCD_p,
                growth_rate = FCD_gr, n_layers_per_block = FCD_l,
                dropout_p = drop):
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
                                    kernel_initializer = initializer
                                    )(inputz)
        n_filters = n_filters_first_conv
        
        # downsampling path, n*(dense block + transition down)
        skip_connection_list = []
        
        for i in range(n_pool):
            ## dense block
            for j in range(n_layers_per_block[i]):
                ### Compute new feature maps
                l = BN_ReLU_Conv(Tiramisu, growth_rate, dropout_p = dropout_p)
                ### and stack it---the Tiramisu is growing
                Tiramisu = ks.layers.concatenate([Tiramisu, l])
                n_filters += growth_rate
            ## store Tiramisu in skip_connections list
            skip_connection_list.append(Tiramisu)
            ## transition Down
            Tiramisu = TransitionDown(Tiramisu, n_filters, dropout_p)
        skip_connection_list = skip_connection_list[::-1]
        
        # bottleneck
        ## store output of subsequent dense block;
        ## upsample only these new features
        block_to_upsample = []
        # dense Block
        for j in range(n_layers_per_block[n_pool]):
            l = BN_ReLU_Conv(Tiramisu, growth_rate, dropout_p = dropout_p)
            block_to_upsample.append(l)
            Tiramisu = ks.layers.concatenate([Tiramisu, l])
        
        # upsampling path
        for i in range(n_pool):
            ## Transition Up ( Upsampling + concatenation with the skip
            ## connection)
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
            outputz = ks.layers.Conv2D(n_classes, (1, 1), \
                                   activation = "softmax")(Tiramisu)
        else:
            outputz = ks.layers.Conv2D(1, (1, 1), \
                                   activation = "sigmoid")(Tiramisu)
        model = tf.keras.Model(inputs = inputz, outputs = outputz)
        print(model.summary())
        print(f'Total number of ks.layers: {len(model.layers)}')
        return model
    
    # get model
    model = FCDense(n_classes = N_CLASSES)
    
    #-------------------------------------------------------------------------
    # DeepLab V3
elif mod == "mod_DL3":
    # https://keras.io/examples/vision/deeplabv3_plus/
    if kernel_init is None:
        initializer = ks.initializers.HeNormal() # or "he_normal"
    else:
        initializer = kernel_init
    def convolution_block(
        block_input,
        num_filters = 256,
        kernel_size = 3,
        dilation_rate = 1,
        padding = "same",
        use_bias = False,
    ):
        x = ks.layers.Conv2D(
            num_filters,
            kernel_size = kernel_size,
            dilation_rate = dilation_rate,
            padding = "same",
            use_bias = use_bias,
            kernel_initializer = initializer,
        )(block_input)
        x = ks.layers.BatchNormalization()(x)
        return tf.nn.relu(x)
    
    def DilatedSpatialPyramidPooling(dspp_input):
        dims = dspp_input.shape
        x = ks.layers.AveragePooling2D(pool_size = (dims[-3], dims[-2]) \
                                    )(dspp_input)
        x = convolution_block(x, kernel_size = 1, use_bias = True)
        out_pool = ks.layers.UpSampling2D(
            size = (dims[-3] // x.shape[1], dims[-2] // x.shape[2]), \
                interpolation = "bilinear",
        )(x)
    
        out_1 = convolution_block(dspp_input, kernel_size = 1, \
                                  dilation_rate = 1)
        out_6 = convolution_block(dspp_input, kernel_size = 3, \
                                  dilation_rate = 6)
        out_12 = convolution_block(dspp_input, kernel_size = 3, \
                                   dilation_rate = 12)
        out_18 = convolution_block(dspp_input, kernel_size = 3, \
                                   dilation_rate = 18)
    
        x = ks.layers.Concatenate(axis = -1)([out_pool, \
                                           out_1, out_6, out_12, out_18])
        output = convolution_block(x, kernel_size = 1)
        return output

    def DeeplabV3Plus(image_size, n_classes):
        inputz = ks.Input(shape = (image_size, image_size, 3))
        resnet50 = ks.applications.ResNet50(
            weights = "imagenet", include_top = False, \
                input_tensor = inputz
        )
        x = resnet50.get_layer("conv4_block6_2_relu").output
        x = DilatedSpatialPyramidPooling(x)
    
        input_a = ks.layers.UpSampling2D(
            size = \
                (image_size // 4 // x.shape[1], image_size // 4 // x.shape[2]),
            interpolation = "bilinear",
        )(x)
        input_b = resnet50.get_layer("conv2_block3_2_relu").output
        input_b = convolution_block(input_b, num_filters = 48, kernel_size = 1)
    
        x = ks.layers.Concatenate(axis=-1)([input_a, input_b])
        x = convolution_block(x)
        x = convolution_block(x)
        x = ks.layers.UpSampling2D(
            size = (image_size // x.shape[1], image_size // x.shape[2]),
            interpolation = "bilinear",
        )(x)
        outputz = ks.layers.Conv2D(n_classes, kernel_size = (1, 1), \
                                     padding = "same")(x)
        return ks.Model(inputs = inputz, outputs = outputz)
    
    # get model
    model = DeeplabV3Plus(image_size = imgr, n_classes = N_CLASSES)

### logs and callbacks---------------------------------------------------------
# define callbacks
from tensorflow.keras.callbacks import LearningRateScheduler
'''
Simple custom LR decay which would only require the epoch index as an argument:
'''
def step_decay_schedule(initial_lr = init_lr,
                        decay_factor = decay_lr, step_size = step_lr):
    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch/step_size))
    
    return LearningRateScheduler(schedule)
#lr_sched = step_decay_schedule(initial_lr = init_lr,
#                               decay_factor = decay_lr, step_size = step_lr)
'''
Using some simple built-in learning rate decay:
'''
if decay_lr < 1.0:
    lr_sched = ks.optimizers.schedules.ExponentialDecay(
    initial_learning_rate = init_lr,
    # decay after n steps
    decay_steps = np.floor(N_img/bs),
    decay_rate = decay_lr)
    optimizers = {
        "adam" : ks.optimizers.Adam(learning_rate = lr_sched, \
                                    clipnorm = 1), \
        "sgd" : ks.optimizers.SGD(learning_rate = lr_sched, \
                                  clipnorm = 1), \
        "rms" : ks.optimizers.RMSprop(learning_rate = lr_sched, \
                                      clipnorm = 1)
        }
elif init_lr is not None:
    optimizers = {
        "adam" : ks.optimizers.Adam(learning_rate = init_lr),
        "sgd" : ks.optimizers.SGD(learning_rate = init_lr),
        "rms" : ks.optimizers.RMSprop(learning_rate = init_lr)
        }
else:
    optimizers = {
        "adam" : ks.optimizers.Adam(),
        "sgd" : ks.optimizers.SGD(),
        "rms" : ks.optimizers.RMSprop()
        }
try:
    optimizer = optimizers[optmer]
except:
    print("Failed to assign optimizer: " + optmer + \
          ". Use 'Adam', 'rms', or 'sgd'.")

# create output directory------------------------------------------------------
os.makedirs(dir_out(), exist_ok = True)

# create log directory---------------------------------------------------------
now = datetime.datetime.now()
logdir = os.path.join(dir_out("logs"), now.strftime("%y-%m-%d-%H-%M-%S"))
cptdir = os.path.join(dir_out("cpts"), now.strftime("%y-%m-%d-%H-%M-%S"))
os.makedirs(logdir, exist_ok = True)
os.makedirs(cptdir, exist_ok = True)

# compile model----------------------------------------------------------------
## loss functions
### define IoU loss (only binary)
#### https://www.youtube.com/watch?v=NqDBvUPD9jg&ab_channel=DigitalSreeni
#def IoU_coe(y_true, y_pred):
#    T = ks.flatten(y_true)
#    P = ks.flatten(y_pred)
#    intersect = ks.sum(T * P)
#    IoU = (intersect + 1.0) / (ks.sum(T) + ks.sum(P) - intersect + 1.0)
#    return IoU

#def IoU_loss(y_true, y_pred):
#    return 1 - IoU_coe(y_true, y_pred)

### define dice coefficient
# https://github.com/tensorlayer/tensorlayer/blob/master/tensorlayer/cost.py#L216
def dice_coe(target, output, loss_type = "jaccard",
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
    return dice
def dice_loss(y_true, y_pred):
    return 1 - dice_coe(y_true, y_pred)

## alternative approach to weighted scce
# https://github.com/tensorflow/models/blob/master/official/nlp/modeling/losses/weighted_sparse_categorical_crossentropy.py

## metrics
# mIoU = ks.metrics.MeanIoU(n_classes = N_CLASSES)
#@ks.utils.register_keras_serializable(package = "Custom", \
#                                      name = "MulticlassMeanIoU")
class SparseMeanIoU(ks.metrics.MeanIoU):
    def __init__(self,
                 y_true = None,
                 y_pred = None,
                 num_classes = None,
                 name = "Sparse_MeanIoU",
                 dtype = None):
        super(SparseMeanIoU, self).__init__(num_classes = num_classes,
                                             name = name, dtype = dtype)
        self.__name__ = "Sparse_MeanIoU"
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

if mk == "f1":
    met = tfa.metrics.F1Score(num_classes = N_CLASSES, threshold = 0.5)
    print("Using metric f1-score")
else:
    met = SparseMeanIoU(num_classes = N_CLASSES)
    print("Using metric mIoU")

metrix = [met, "sparse_categorical_accuracy"] if N_CLASSES > 2 else \
    [met, "accuracy"]

### get sparse categorical/binary cross entropy
lozz = ks.losses.SparseCategoricalCrossentropy() if N_CLASSES > 2 else\
    ks.losses.BinaryCrossentropy()

#run_opts = tf.compat.v1.RunOptions(report_tensor_allocations_upon_oom = True)

# callbacks--------------------------------------------------------------------
# monitor epoch time callback
class TimestampCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.metric_name = "EpochTime"
        self.__epoch_start = None
    def on_epoch_begin(self, epoch, logs = None):
        self.__epoch_start = tf.timestamp()
    def on_epoch_end(self, epoch, logs = None):
        logs[self.metric_name] = (tf.timestamp() - \
            self.__epoch_start).numpy()

monitor_metric = cb_metric if cb_metric is not None else met.name
cllbs = [
    ks.callbacks.ModelCheckpoint(os.path.join(cptdir, \
                                              "Epoch.{epoch:02d}.hdf5"), \
                                 monitor = monitor_metric, \
                                 mode = monitor_mode, \
                                 save_best_only = False, \
                                 save_freq = "epoch"),
    ks.callbacks.TensorBoard(log_dir = logdir, histogram_freq = 5),
    TimestampCallback()
    ]
if es_patience is not None:
    cllbs.append(ks.callbacks.EarlyStopping(monitor = monitor_metric, \
                                            mode = monitor_mode, \
                               patience = es_patience, verbose = 1))
rop = None
if rop is not None:
    cllbs.append(ks.callbacks.ReduceLROnPlateau(monitor = monitor_metric, \
                                                factor = rop, patience = 5, \
                                                    min_lr = 1e-5))

# resume training or compile new model-----------------------------------------
if resume_training == "f":
    os.chdir(logdir)
    model.compile(optimizer = optimizer, loss = lozz,
                  metrics = metrix)#, options = run_opts)
    model.summary()
elif resume_training == "t":
    cpt_folders = [f for f in os.listdir(dir_out("cpts")) \
                   if not (f.startswith(".") or \
                           os.listdir(dir_out("cpts", f))==[])]
    cpt_dates = [datetime.datetime.strptime(d, "%y-%m-%d-%H-%M-%S"\
                                            ) for d in cpt_folders]
    cpt_folder = max(cpt_dates).strftime("%y-%m-%d-%H-%M-%S")
    debug_cp(line = "Resume training from " + cpt_folder)
else:
    cpt_folder = resume_training

if resume_training != "f":
    if resume_training == "t":
        list_of_files = glob.glob(dir_out("cpts", cpt_folder) + os.path.sep + \
                                  "*" + ".hdf5")
        trained_model = glob.glob(dir_out("cpts", "trained_mod", \
                                          "saved_model.pb"))
        model_folder = [os.path.dirname(trained_model[0])] if \
                        len(trained_model) >= 1 else []
        list_of_files += model_folder
    elif os.path.isfile(cpt_folder):
        list_of_files = [os.path.dirname(cpt_folder)] if ".pb" in cpt_folder \
            else [cpt_folder]
    else:
        list_of_files = glob.glob(cpt_folder + os.path.sep + \
                                  "*" + ".hdf5")
        trained_model = glob.glob(os.path.join(cpt_folder, "saved_model.pb"))
        model_folder = [os.path.dirname(trained_model[0])] if \
                        len(trained_model) >= 1 else []
        list_of_files += model_folder

    checkpoint = max(list_of_files, key = os.path.getctime)
    #try:
    co = {"SparseMeanIoU": met} if met.name == "Sparse_MeanIoU" else \
        {"F1": met}
    model = ks.models.load_model(checkpoint, \
                                 custom_objects = co)
    #except:
    #    print("Failed to load model from", checkpoint)
    all_logs = [dir_out("logs", p) for p in os.listdir(dir_out("logs"))]
    logdir =  max(all_logs, key = os.path.getctime)
    os.chdir(logdir)
    '''
    model.compile(optimizer = optimizer, loss = lozz,
                  metrics = metrix)
    '''

# report to tensorboard--------------------------------------------------------
if tb:
    debug_cp(line = "Start Tensorboard dev. Directory: " + logdir + "\n")
    import subprocess
    PARAMETERS = "Batch size: " + str(bs) + " Init. lr: " + str(init_lr) + \
        " Img dim: " + str(imgc) + " Weights: " + str(ww) + " Optimizer: " + \
            optmer + " Dataset: " + year
    '''
    not working:
    subprocess.Popen(["tensorboard", "dev", "upload --logdir '" + logdir + \
                      "' --name LeleNet_" + mod + " --description '" + \
                          PARAMETERS + "'"])
    tb_str = "tensorboard dev upload --logdir '" + logdir + \
                      "' --name LeleNet_" + mod + " --description '" + \
                          PARAMETERS + "'"
    '''
    if tbn is None:
        tbn = "LeleNet_" + mod + "_" + year
    subprocess.call("tensorboard dev upload --logdir '" + logdir + \
                    "' --name " + tbn + " --description '" + \
                    PARAMETERS + "' &", shell = True)

# fit model--------------------------------------------------------------------
debug_cp(line = "Fit model...\n")
args_fit = {"epochs" : epochz,
            "steps_per_epoch" : np.ceil(N_img/bs),
            "validation_steps" : np.ceil(N_val/bs),
            "callbacks" : cllbs}
if resume_training != "f":
    try:
        s = checkpoint.find("Epoch.") + len("Epoch.")
        e = checkpoint.find(".hdf5")
        args_fit["initial_epoch"] = int(checkpoint[s : e])
    except:
        print("Error when trying to retreive the epoch number from filename",\
              "'" + checkpoint + "': Unable to find integer at position", \
                  str(checkpoint.find("Epoch.") + len("Epoch.")), "to", \
                      str(len(checkpoint)-5))

if "train_generator" in locals() or "train_generator" in globals():
    '''
    model.fit(train_generator,
                     validation_data = val_generator,
                     **args_fit)
    '''
    Warning("Currently no Keras ImageDataGenerators supported.")
else:
    if ww != 0.0:
        hist = model.fit(dataset["train"].map(add_sample_weights),
                         validation_data = dataset["val"],
                         **args_fit)
    else:
        hist = model.fit(dataset["train"],
                         validation_data = dataset["val"],
                         **args_fit)

os.chdir(dir_out())

# save model-------------------------------------------------------------------
model.save(dir_out("cpts", "trained_mod"), save_format = "tf", \
           save_traces = True)
#model.save(dir_out("cpts", "trained_mod", "model.h5"), save_format = "h5")
print("Model saved to disc at " + dir_out("cpts", "trained_mod"))

# save custom metrics----------------------------------------------------------
with open(dir_out("cpts", "trained_mod", "custom_metrics.pkl"), "wb") as f:
    pickle.dump(met, f)

# save model history to csv----------------------------------------------------
import pandas as pd
hist_df = pd.DataFrame(hist.history)
hist_dir = dir_out("hist", now.strftime("%y-%m-%d-%H-%M-%S"))
os.makedirs(hist_dir, exist_ok = True)
hist_csv_path = os.path.join(hist_dir, "history.csv")
with open(hist_csv_path, mode = "w") as f:
    hist_df.to_csv(f)

print("Task completed:", datetime.datetime.utcnow())
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 17 17:56:34 2021

@author: Manuel
Set general functions and variables.
"""
__author__ = "Manuel R. Popp"

#### basic settings------------------------------------------------------------
import platform, sys, datetime
OS = platform.system()
OS_version = platform.release()
py_version = sys.version
t_start = datetime.datetime.utcnow()
print("Running on " + OS + " " + OS_version + ".\nPython version: " +
      py_version + "\nUTC time (start): " + str(t_start) +
      "\nLocal time (start): " + str(datetime.datetime.now()))
# Image formats
xf = "png"
yf = "png"
imgr = 512
imgc = 512
imgdim = 3

# Model (one of "mod_UNet", "mod_FCD")
mod = "mod_UNet"

### general directory functions------------------------------------------------
import os
import numpy as np
if OS == "Linux":
    wd = "/home/manuel/Nextcloud/Masterarbeit"
elif OS == "Windows":
    wd = os.path.join("C:\\", "Users", "Manuel", "Nextcloud", "Masterarbeit")
else:
    raise Exception("OS not detected.")

def dir_fig(fig_id = None):
    if fig_id == None:
        return os.path.join(wd, "fig")
    else:
        return os.path.join(wd, "fig", fig_id)

def dir_dat(dat_id = None):
    if dat_id == None:
        return os.path.join(wd, "dat")
    else:
        return os.path.join(wd, "dat", dat_id)

def dir_out(out_id = None):
    if out_id == None:
        return os.path.join(wd, "out")
    else:
        return os.path.join(wd, "out", out_id)

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

#### data preparation directory functions--------------------------------------
import pathlib
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

def dir_tls(plot_id = None, myear = None, dset = None):
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
                raise Exception("Missing dset (X or y). Returning tile directory.")
            else:
                return os.path.join(dir_dat("tls"), myear, dset, "0", plot_id)
def toINT(filename):
    imgINT = filename.astype("uint8")
    return imgINT

# Data preparation-------------------------------------------------------------
## select year for data preparation
year = "03_2021"

### Run file DataPreparation.py
# exec(open("A1_DataPreparation.py").read())

## load information generated during data preparation--------------------------
classes, classes_decoded, NoDataValue, no_data_class = get_var("ClassIDs")
N_CLASSES = len(classes) if no_data_class else len(classes) + 1

# Setup for training-----------------------------------------------------------
os.chdir(os.path.join(wd, "py3"))
exec(open("A2_PreTrainSetup_CustomDL.py").read())

# Get model--------------------------------------------------------------------
os.chdir(os.path.join(wd, "py3"))
if mod == "mod_UNet":
    exec(open("A4_Models_UNet.py").read())
elif mod == "FCD":
    exec(open("A4_Models_FCDenseNet.py").read())

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
lr_sched = step_decay_schedule(initial_lr = 1e-3,
                               decay_factor = 0.995, step_size = 2)
'''
Using some simple built-in learning rate decay:
'''
lr_sched = ks.optimizers.schedules.ExponentialDecay(
    initial_learning_rate = 1e-3,
    # decay after n steps
    decay_steps = np.floor(N_img/bs),
    decay_rate = 0.995)

optimizer = ks.optimizers.Adam(learning_rate = lr_sched, clipnorm = 1)
optimizer = ks.optimizers.SGD(learning_rate = 0.0001, clipnorm = 1)
optimizer = ks.optimizers.RMSprop(learning_rate = lr_sched, clipnorm = 1)# FC-DenseNet Optim.

# list callbacks
logdir = os.path.join(dir_out("logs"), datetime.datetime.now() \
                      .strftime("%y-%m-%d-%H-%M-%S"))
os.makedirs(logdir)
os.chdir(logdir)
cllbs = [
    ks.callbacks.ReduceLROnPlateau(monitor = "val_loss", factor = 0.2,
                                   patience = 5, min_lr = 0.001),
    ks.callbacks.EarlyStopping(patience = 8),
    ks.callbacks.ModelCheckpoint(dir_out("Checkpoint.h5"),
                                 save_best_only = True),
    ks.callbacks.TensorBoard(log_dir = logdir)
    ]

# compile model----------------------------------------------------------------
lozz = ks.losses.SparseCategoricalCrossentropy()

#lozz = wcc_loss
model.compile(optimizer = optimizer, loss = lozz,
              metrics = ["accuracy", "sparse_categorical_accuracy"])
model.summary()
# fit model--------------------------------------------------------------------
if "train_generator" in locals() or "train_generator" in globals():
    model.fit(train_generator, epochs = 45, steps_per_epoch = np.ceil(N_img/bs),
                     validation_data = val_generator,
                     validation_steps = np.ceil(N_val/bs),
                     callbacks = cllbs)
else:     
    model.fit(dataset["train"].map(add_sample_weights),
                     epochs = 45, steps_per_epoch = np.ceil(N_img/bs),
                     validation_data = dataset["val"],
                     validation_steps = np.ceil(N_val/bs),
                     callbacks = cllbs)
os.chdir(dir_out())
# save model-------------------------------------------------------------------
os.makedirs(dir_out("mod"), exist_ok = True)
model.save(dir_out(mod))

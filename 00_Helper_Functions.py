#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 17 17:56:34 2021

@author: Manuel
Set general functions and variables.
"""
__author__ = "Manuel R. Popp"

#### basic settings
import platform, sys, datetime
OS = platform.system()
OS_version = platform.release()
py_version = sys.version
t_start = datetime.datetime.utcnow()
print("Running on " + OS + " " + OS_version + ".\nPython version: " +
      py_version + "\nUTC time (start): " + str(t_start) +
      "\nLocal time (start): " + str(datetime.datetime.now()))

### directories
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

#### generate tiles and masks
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

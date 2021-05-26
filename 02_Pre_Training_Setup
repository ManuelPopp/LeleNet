#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 17 17:56:34 2021

@author: Manuel
Create ImageDataGenerator and define options for training the CNN.
"""
__author__ = "Manuel R. Popp"

os.chdir(wd)

classes, classes_decoded, NoDataValue = get_var("ClassIDs")
#### semantic segmentation
import tensorflow as tf
from tensorflow import keras as ks
tf.__version__
ks.__version__

### make GPU available
phys_devs = tf.config.experimental.list_physical_devices("GPU")
print("N GPUs available: ", len(phys_devs))
if len(phys_devs) >= 1 and False:
    tf.config.experimental.set_memory_growth(phys_devs[0], True)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

### build data generators
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
            "width_shift_range" : 0.05,
            "height_shift_range" : 0.05,
            "horizontal_flip" : True,
            "vertical_flip" : True,
            "fill_mode" : "constant",
           # "featurewise_std_normalization" : False,
            "featurewise_center" : False
            }
args_flow = {"class_mode" : None,
             "batch_size" : bs,
             "target_size" : (imgr, imgc),
             "seed" : zeed
             }

X_generator = ks.preprocessing.image.ImageDataGenerator(rescale = 1.0/255.0,
                                                        **args_aug,
                                                        **args_col)
#X_generator.fit(dir_tls(myear = year, dset = "X"))
X_gen = X_generator.flow_from_directory(directory = os.path.dirname(
                        dir_tls(myear = year, dset = "X")),
                        color_mode = "rgb",
                        **args_flow)

y_generator = ks.preprocessing.image.ImageDataGenerator(**args_aug,
                                                        cval = NoDataValue)
y_gen = y_generator.flow_from_directory(directory = os.path.dirname(
                        dir_tls(myear = year, dset = "y")),
                        color_mode = "grayscale",
                        interpolation = "nearest",
                        **args_flow)

train_generator = zip(X_gen, y_gen)
# val generator
X_val_generator = ks.preprocessing.image.ImageDataGenerator(rescale = 1.0/255.0)
X_val_gen = X_generator.flow_from_directory(directory = os.path.dirname(
                        dir_tls(myear = year, dset = "X_val")),
                        color_mode = "rgb",
                        **args_flow)

y_val_generator = ks.preprocessing.image.ImageDataGenerator()
y_val_gen = y_generator.flow_from_directory(directory = os.path.dirname(
                        dir_tls(myear = year, dset = "y_val")),
                        color_mode = "grayscale",
                        interpolation = "nearest",
                        **args_flow)

val_generator = zip(X_val_gen, y_val_gen)
### logs and callbacks
os.chdir(dir_out())# write logs to out dir
# define callbacks
from tensorflow.keras.callbacks import LearningRateScheduler

def step_decay_schedule(initial_lr = 1e-3,
                        decay_factor = 0.75, step_size = 10):
    '''
    Wrapper function to create a LearningRateScheduler with step decay schedule.
    '''
    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch/step_size))
    
    return LearningRateScheduler(schedule)

lr_sched = step_decay_schedule(initial_lr = 1e-4,
                               decay_factor = 0.75, step_size = 2)
# list callbacks
cllbs = [
    ks.callbacks.EarlyStopping(patience = 8),
    ks.callbacks.ModelCheckpoint(dir_out("Checkpoint.h5"),
                                 save_best_only = True),
    lr_sched,
    ks.callbacks.TensorBoard(log_dir = dir_out("logs"))
    ]

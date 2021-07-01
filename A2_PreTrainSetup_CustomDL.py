#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 23:02:51 2021

@author: Manuel
Create ImageDataGenerator and define options for training the CNN.
"""
__author__ = "Manuel R. Popp"

os.chdir(wd)

# import modules---------------------------------------------------------------
import tensorflow as tf
import tensorflow_io as tfio
from tensorflow import keras as ks
AUTOTUNE = tf.data.experimental.AUTOTUNE
tf.__version__
ks.__version__

## make GPU available----------------------------------------------------------
phys_devs = tf.config.experimental.list_physical_devices("GPU")
print("N GPUs available: ", len(phys_devs))
if len(phys_devs) >= 1 and False:
    tf.config.experimental.set_memory_growth(phys_devs[0], True)
else:
    #os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
    my_devices = tf.config.experimental.list_physical_devices(device_type = "CPU")
    tf.config.experimental.set_visible_devices(devices = my_devices, device_type = "CPU")
    print("No GPUs used.")

## general options/info--------------------------------------------------------
N_img = len(list(pathlib.Path(dir_tls(myear = year, dset = "X")) \
                 .glob("**/*." + xf)))
N_val = len(list(pathlib.Path(dir_tls(myear = year, dset = "X_val")) \
                 .glob("**/*." + xf)))
bs = 1
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

# define weighted categorical crossentropy loss function-----------------------
from PIL import Image
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
        classweights[unique.astype(int) - 1] = counts
        weights = ((weights * gravity) + classweights) / (gravity + 1)
        gravity += 1
    return weights

WEIGHTS = calculate_weights(os.path.dirname(dir_tls(myear = year, dset = "y")),
                     N_CLASSES)
NORMWEIGHTS = WEIGHTS / max(WEIGHTS)
### inverse frequency as weights
#inv_weights = tf.constant((1 / (WEIGHTS + 0.01)), dtype = tf.float32,
#                          shape = [1, 1, 1, N_CLASSES])
inv_weights = 1 / (NORMWEIGHTS + 0.01)

## add weights-----------------------------------------------------------------
def add_sample_weights(image, segmentation_mask):
  class_weights = tf.constant(inv_weights, dtype = tf.float32)
  class_weights = class_weights/tf.reduce_sum(class_weights)
  sample_weights = tf.gather(class_weights,
                             indices = tf.cast(segmentation_mask, tf.int32))
  return image, segmentation_mask, sample_weights

dataset["train"].map(add_sample_weights).element_spec

##############################################################################
def wcc_loss(y_true, y_pred, n_classes = N_CLASSES, w = inv_weights):
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
    y_true = tf.cast(y_true, tf.int32)
    if len(y_pred.shape) == len(y_true.shape):
        y_true = tf.squeeze(y_true, [-1])
    w = tf.cast(w, y_pred.dtype)
    # calculate loss
    #wloss = tf.squeeze(onehot) * tf.keras.backend.log(y_pred) * w
    scce_loss = ks.losses.SparseCategoricalCrossentropy(y_true, y_pred)
    wloss = tf.math.divide_no_nan(tf.reduce_sum(scce_loss * w),
                                  tf.reduce_sum(w))
    return wloss
# -*- coding: utf-8 -*-
"""
Created on Sat May 29 23:36:08 2021

@author: Manuel

Implementation of the U-Net structure proposed by Ronneberger et al. (2015)
in the following paper https://www.doi.org/10.1007/978-3-319-24574-4_28
in tf.Keras.
"""

def UNet(n_classes, input_shape = (imgr, imgc, imgdim), dropout = 0.5,
         ops = {"activation" : "relu",
                "padding" : "same",
                "kernel_initializer" : "he_normal"
        }):
    # input layer
    inputz = ks.layers.Input(shape = input_shape)
    
    # encoder part
    ## 1st convolution
    c1 = ks.layers.Conv2D(64, (3, 3), **ops)(inputz)
    c1 = ks.layers.Conv2D(64, (3, 3), **ops)(c1)
    ## 1st max pooling
    p1 = ks.layers.MaxPooling2D(pool_size = (2, 2))(c1)
    
    ## 2nd convolution
    c2 = ks.layers.Conv2D(128, (3, 3), **ops)(p1)
    c2 = ks.layers.Conv2D(128, (3, 3), **ops)(c2)
    ## 2nd max pooling
    p2 = ks.layers.MaxPooling2D(pool_size = (2, 2))(c2)
    
    ## 3rd convolution
    c3 = ks.layers.Conv2D(256, (3, 3), **ops)(p2)
    c3 = ks.layers.Conv2D(256, (3, 3), **ops)(c3)
    ## 3rd max pooling
    p3 = ks.layers.MaxPooling2D(pool_size = (2, 2))(c3)
    
    ## 4th convolution
    c4 = ks.layers.Conv2D(512, (3, 3), **ops)(p3)
    c4 = ks.layers.Conv2D(512, (3, 3), **ops)(c4)
    ## Drop
    d4 = ks.layers.Dropout(dropout)(c4)
    ## 4th max pooling
    p4 = ks.layers.MaxPooling2D(pool_size = (2, 2))(d4)
    
    ## 5th convolution
    c5 = ks.layers.Conv2D(1024, (3, 3), **ops)(p4)
    c5 = ks.layers.Conv2D(1024, (3, 3), **ops)(c5)
    ## Drop
    d5 = ks.layers.Dropout(dropout)(c5)
    
    # decoder part
    ## 1st up convolution
    us6 = ks.layers.UpSampling2D(size = (2, 2))(d5)
    up6 = ks.layers.Conv2D(512, (2, 2), **ops)(us6)
    ## merge
    ct6 = ks.layers.concatenate([d4, up6], axis = 3)
    uc6 = ks.layers.Conv2D(512, (3, 3), **ops)(ct6)
    uc6 = ks.layers.Conv2D(512, (3, 3), **ops)(uc6)
    
    ## 2nd up convolution
    us7 = ks.layers.UpSampling2D(size = (2, 2))(uc6)
    up7 = ks.layers.Conv2D(256, (2, 2), **ops)(us7)
    ## merge
    ct7 = ks.layers.concatenate([c3, up7], axis = 3)
    uc7 = ks.layers.Conv2D(256, (3, 3), **ops)(ct7)
    uc7 = ks.layers.Conv2D(256, (2, 2), **ops)(uc7)
     
    ## 3rd up convolution
    us8 = ks.layers.UpSampling2D(size = (2, 2))(uc7)
    up8 = ks.layers.Conv2D(128, (2, 2), **ops)(us8)
    ## merge
    ct8 = ks.layers.concatenate([c2, up8], axis = 3)
    uc8 = ks.layers.Conv2D(128, (3, 3), **ops)(ct8)
    uc8 = ks.layers.Conv2D(128, (3, 3), **ops)(uc8)
     
    ## 4th up convolution
    us9 = ks.layers.UpSampling2D(size = (2, 2))(uc8)
    up9 = ks.layers.Conv2D(64, (2, 2), **ops)(us9)
    ## merge
    ct9 = ks.layers.concatenate([c1, up9], axis = 3)
    uc9 = ks.layers.Conv2D(64, (3, 3), **ops)(ct9)
    uc9 = ks.layers.Conv2D(64, (3, 3), **ops)(uc9)
    uc9 = ks.layers.Conv2D(2, (3, 3), **ops)(uc9)
    
    # output layer
    outputz = ks.layers.Conv2D(n_classes, 1, activation = "softmax")(uc9)
    
    model = ks.Model(inputs = [inputz], outputs = [outputz])
    print(model.summary())
    print(f'Total number of layers: {len(model.layers)}')
    return model

# get model
model = UNet(n_classes = N_CLASSES)

# directory to save model
os.makedirs(dir_out("mod_UNet"), exist_ok = True)

# -*- coding: utf-8 -*-
"""
Created on Sat May 29 18:48:59 2021

@author: Manuel
This code implements the Fully Convolutional DenseNet described in
https://arxiv.org/abs/1611.09326 using tf.Keras.
The original implementation using Theano and Lasagne was made public on
https://github.com/SimJeg/FC-DenseNet/blob/master/FC-DenseNet.py
When using this as standalone script, import the following hashed-out packages
"""
# import tensorflow as tf
# from tensorflow import keras as ks

# Define blocks
def BN_ReLU_Conv(inputs, n_filters, filter_size = 3, dropout_p = 0.2):
    l = ks.layers.BatchNormalization()(inputs)
    l = ks.layers.Activation("relu")(l)
    l = ks.layers.Conv2D(n_filters, filter_size, activation = None, padding = "same", 
                         kernel_initializer = 'he_uniform') (l)
    if dropout_p != 0.0:
        l = ks.layers.Dropout(dropout_p)(l)
    return l

def TransitionDown(inputs, n_filters, dropout_p = 0.2):
    l = BN_ReLU_Conv(inputs, n_filters, filter_size = 1, dropout_p = dropout_p)
    l = ks.layers.MaxPool2D(pool_size = (2, 2))(l)
    return l

def TransitionUp(skip_connection, block_to_upsample, n_filters_keep):
    l = ks.layers.concatenate(block_to_upsample)
    l = ks.layers.Conv2DTranspose(n_filters_keep, kernel_size = (3, 3),
                                  strides = (2, 2), padding = "same", 
                                  kernel_initializer = "he_uniform") (l)
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
    Skip connections are used between the downsampling path and the upsampling
    path
    Each layer is a composite function of BN - ReLU - Conv and the last layer
    is a softmax layer.
    :param input_shape: shape of the input batch. Only the first dimension
        (n_channels) is needed
    :param n_classes: number of classes
    :param n_filters_first_conv: number of filters for the first convolution
        applied
    :param n_pool: number of pooling layers = number of transition down =
        number of transition up
    :param growth_rate: number of new feature maps created by each layer in a
        dense block
    :param n_layers_per_block: number of layers per block. Can be an int or a
        list of size 2 * n_pool + 1
    :param dropout_p: dropout rate applied after each convolution
        (0. for not using)
    """
    # Check n_layers_per_block setting
    if type(n_layers_per_block) == list:
        assert(len(n_layers_per_block) == 2*n_pool + 1)
    elif type(n_layers_per_block) == int:
        n_layers_per_block = [n_layers_per_block]*(2*n_pool + 1)
    else:
        raise ValueError
    # Input layer, m = 3
    inputz = tf.keras.layers.Input(shape = input_shape)
    
    # First convolution; store feature maps in the Tiramisu
    # 3 x 3 convolution, m = 48
    Tiramisu = ks.layers.Conv2D(filters = n_filters_first_conv,
                                kernel_size = (3, 3), strides = (1, 1),
                                padding = "same", dilation_rate = (1, 1),
                                activation = "relu",
                                kernel_initializer = "he_uniform"
                                )(inputz)
    n_filters = n_filters_first_conv
    
    # Downsampling path, n*(dense block + transition down)
    skip_connection_list = []
    
    for i in range(n_pool):
        ## Dense block
        for j in range(n_layers_per_block[i]):
            ### Compute new feature maps
            l = BN_ReLU_Conv(Tiramisu, growth_rate, dropout_p=dropout_p)
            ### And stack it---the Tiramisu is growing
            Tiramisu = ks.layers.concatenate([Tiramisu, l])
            n_filters += growth_rate
        ## Store Tiramisu in skip_connections list
        skip_connection_list.append(Tiramisu)
        ## Transition Down
        Tiramisu = TransitionDown(Tiramisu, n_filters, dropout_p)
    skip_connection_list = skip_connection_list[::-1]
    
    # Bottleneck
    ## Store output of subsequent dense block; upsample only these new features
    block_to_upsample = []
    # Dense Block
    for j in range(n_layers_per_block[n_pool]):
        l = BN_ReLU_Conv(Tiramisu, growth_rate, dropout_p = dropout_p)
        block_to_upsample.append(l)
        Tiramisu = ks.layers.concatenate([Tiramisu, l])
    
    # Upsampling path
    for i in range(n_pool):
        ## Transition Up ( Upsampling + concatenation with the skip connection)
        n_filters_keep = growth_rate * n_layers_per_block[n_pool + i]
        Tiramisu = TransitionUp(skip_connection_list[i], block_to_upsample,
                             n_filters_keep)
        ## Dense Block
        block_to_upsample = []
        for j in range(n_layers_per_block[n_pool + i + 1]):
            l = BN_ReLU_Conv(Tiramisu, growth_rate, dropout_p = dropout_p)
            block_to_upsample.append(l)
            Tiramisu = ks.layers.concatenate([Tiramisu, l])
    
    # Output layer; 1x1 convolution, m = number of classes
    outputz = ks.layers.Conv2D(n_classes, 1, activation = "softmax")(Tiramisu)
    
    model = tf.keras.Model(inputs = [inputz], outputs = [outputz])
    print(model.summary())
    print(f'Total number of layers: {len(model.layers)}')
    return model

# Get model
model = FCDense(n_classes = N_CLASSES)

# Using some simple built-in learning rate decay:
lr_sched = ks.optimizers.schedules.ExponentialDecay(
    initial_learning_rate = 1e-3,
    # decay after n steps
    decay_steps = np.floor(N_img/bs),
    decay_rate = 0.995)
optimizer = ks.optimizers.RMSprop(learning_rate = lr_sched)# FC-DenseNet Optim.

# list callbacks
logdir = os.path.join(dir_out("logs"), datetime.datetime.now() \
                      .strftime("%y-%m-%d-%H-%M-%S"))
os.makedirs(logdir)
os.chdir(logdir)
cllbs = [
    ks.callbacks.EarlyStopping(patience = 8),
    ks.callbacks.ModelCheckpoint(dir_out("Checkpoint.h5"),
                                 save_best_only = True),
    ks.callbacks.TensorBoard(log_dir = logdir)
    ]
# compile model
model.compile(optimizer = optimizer, loss = "sparse_categorical_crossentropy",
              metrics = ["accuracy"])
model.summary()

model.fit(train_generator, epochs = 45, steps_per_epoch = np.ceil(N_img/bs),
                 validation_data = val_generator,
                 validation_steps = np.ceil(N_val/bs),
                 callbacks = cllbs)
os.chdir(dir_out())
# save model
os.makedirs(dir_out("mod_FCD"), exist_ok = True)
model.save(dir_out("mod_FCD"))

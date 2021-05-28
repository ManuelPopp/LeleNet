from tensorflow.keras.callbacks import LearningRateScheduler
'''
Simple custom LR decay which would only require the epoch index as an argument:
'''
def step_decay_schedule(initial_lr = 1e-3,
                        decay_factor = 0.75, step_size = 10):
    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch/step_size))
    
    return LearningRateScheduler(schedule)
lr_sched = step_decay_schedule(initial_lr = 1e-4,
                               decay_factor = 0.75, step_size = 2)
'''
Using some simple built-in learning rate decay:
'''
lr_sched = ks.optimizers.schedules.ExponentialDecay(
    initial_learning_rate = 1e-2,
    # decay after n steps
    decay_steps = 1000,
    decay_rate = 0.9)
optimizer = ks.optimizers.Adam(learning_rate = lr_sched)
optimizer = ks.optimizers.SGD(learning_rate = 0.0001)

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
model.compile(optimizer = optimizer, loss = "sparse_categorical_crossentropy",# "weighted_categorical_crossentropy",
              metrics = ["accuracy"])
model.summary()
# fit model
model.fit(train_generator, epochs = 45, steps_per_epoch = np.ceil(N_img/bs),
                 validation_data = val_generator,
                 validation_steps = np.ceil(N_val/bs),
                 callbacks = cllbs)
os.chdir(dir_out())
# save model
os.makedirs(dir_out("mod"), exist_ok = True)
model.save(dir_out("mod"))

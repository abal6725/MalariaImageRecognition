import time
from tensorflow import keras
import math
from tensorflow.keras.callbacks import ModelCheckpoint


# Evaluation Time callback
class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


# MODEL 1
model = keras.Sequential()

model.add(keras.layers.Flatten(input_shape = (64,64,3)))

model.add(keras.layers.Dense(units = 512, activation = 'relu'))

model.add(keras.layers.Dense(units = 128, activation = 'relu'))

model.add(keras.layers.Dense(units = 1, activation = 'sigmoid'))

# load weights
# model.load_weights("weights.best.hdf5")

model.compile(optimizer = 'adam',loss = 'binary_crossentropy', metrics = ['accuracy'])

### Using tensorboard callbacks
tbCallBack = keras.callbacks.TensorBoard(log_dir='./Model1/TBGraph', histogram_freq=0, write_graph=True, write_images=True)

# checkpoint Callback
filepath="./Model1/checkpoint/weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

time_callback = TimeHistory()

callbacks_list = [checkpoint, tbCallBack, time_callback]

model.fit_generator(train_generator, steps_per_epoch=689, epochs = 50,validation_data=validation_generator, callbacks=callbacks_list)

model1_time = sum(time_callback.times)



## MODEL 1 WITH DROPOUT LAYER TO COMBAT OVERFITTING
# rate = 0.25

model = keras.Sequential()

model.add(keras.layers.Flatten(input_shape = (64,64,3)))

model.add(keras.layers.Dropout(rate=0.25))

model.add(keras.layers.Dense(units = 512, activation = 'relu'))

model.add(keras.layers.Dropout(rate=0.25))

model.add(keras.layers.Dense(units = 128, activation = 'relu'))

model.add(keras.layers.Dropout(rate=0.25))

model.add(keras.layers.Dense(units = 1, activation = 'sigmoid'))

# load weights
# model.load_weights("weights.best.hdf5")

model.compile(optimizer = 'adam',loss = 'binary_crossentropy', metrics = ['accuracy'])

### Using tensorboard callbacks
tbCallBack = keras.callbacks.TensorBoard(log_dir='./Model1/Dropout/TBGraph', histogram_freq=0, write_graph=True, write_images=True)

# checkpoint Callback
filepath="./Model1/Dropout/checkpoint/weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

time_callback = TimeHistory()

callbacks_list = [checkpoint, tbCallBack, time_callback]

model.fit_generator(train_generator, steps_per_epoch=689, epochs = 50,validation_data=validation_generator, callbacks=callbacks_list)

model1_drop_time = sum(time_callback.times)


## MODEL 1 WITH BATCHNORMALIZATION

model = keras.Sequential()

model.add(keras.layers.Flatten(input_shape = (64,64,3)))

model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Dense(units = 512, activation = 'relu'))

model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Dense(units = 128, activation = 'relu'))

model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Dense(units = 1, activation = 'sigmoid'))

# load weights
# model.load_weights("weights.best.hdf5")

model.compile(optimizer = 'adam',loss = 'binary_crossentropy', metrics = ['accuracy'])

### Using tensorboard callbacks
tbCallBack = keras.callbacks.TensorBoard(log_dir='./Model1/Norm/TBGraph', histogram_freq=0, write_graph=True, write_images=True)

# checkpoint Callback
filepath="./Model1/Norm/checkpoint/weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

time_callback = TimeHistory()

callbacks_list = [checkpoint, tbCallBack, time_callback]

model.fit_generator(train_generator, steps_per_epoch=689, epochs = 25,validation_data=validation_generator, callbacks=callbacks_list)

model1_norm_time = sum(time_callback.times)



## MODEL 1 WITH DROPOUT + BATCHNORMALIZATION
# rate = 0.25

model = keras.Sequential()

model.add(keras.layers.Flatten(input_shape = (64,64,3)))

model.add(keras.layers.Dropout(rate=0.25))

model.add(keras.layers.Dense(units = 512, activation = 'relu'))

model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Dropout(rate=0.25))

model.add(keras.layers.Dense(units = 128, activation = 'relu'))

model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Dropout(rate=0.25))

model.add(keras.layers.Dense(units = 10, activation = 'softmax'))

# load weights
# model.load_weights("weights.best.hdf5")

model.compile(optimizer = 'adam',loss = 'categorical_crossentropy', metrics = ['accuracy'])

### Using tensorboard callbacks
model_dir ='./CIFAR-10_Models/Model1'
tbCallBack = keras.callbacks.TensorBoard(log_dir= model_dir+'/TBGraph', histogram_freq=0, write_graph=True, write_images=True)

# checkpoint Callback
filepath = model_dir+"/checkpoint/weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

time_callback = TimeHistory()

callbacks_list = [checkpoint, tbCallBack, time_callback]

print('MODEL 1 WITH DROPOUT + BATCH NORMALIZATION')

model.fit_generator(train_generator, steps_per_epoch=train_steps, epochs = 50, validation_data=validation_generator, callbacks=callbacks_list,verbose = 2)

times = time_callback.times


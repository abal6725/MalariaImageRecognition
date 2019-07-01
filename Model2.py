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



####### Model2 ######

model = keras.Sequential()

model.add(keras.layers.Conv2D(filters = 64, kernel_size=(3,3), input_shape = (64,64,3), activation='relu', padding = 'same'))

model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(keras.layers.Conv2D(filters = 128, kernel_size=(3,3), activation='relu', padding = 'same'))

model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(keras.layers.Conv2D(filters = 256, kernel_size=(3,3), activation='relu', padding = 'same'))

model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(units = 512, activation = 'relu'))

model.add(keras.layers.Dense(units = 128, activation = 'relu'))

model.add(keras.layers.Dense(units = 1, activation = 'sigmoid'))

# load weights
# model.load_weights("weights.best.hdf5")

model.compile(optimizer = 'adam',loss = 'binary_crossentropy', metrics = ['accuracy'])

### Using tensorboard callbacks
tbCallBack = keras.callbacks.TensorBoard(log_dir='./Model2/TBGraph', histogram_freq=0, write_graph=True, write_images=True)

# checkpoint Callback
filepath="./Model1/checkpoint/weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

time_callback = TimeHistory()

callbacks_list = [checkpoint, tbCallBack, time_callback]

model.fit_generator(train_generator, steps_per_epoch=689, epochs = 50,validation_data=validation_generator, callbacks=callbacks_list)

model2_time = sum(time_callback.times)



# MODEL 2 WITH DROPOUT
model = keras.Sequential()

model.add(keras.layers.Conv2D(filters = 64, kernel_size=(3,3), input_shape = (64,64,3), activation='relu', padding = 'same'))

model.add(keras.layers.Dropout(rate=0.25))

model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(keras.layers.Conv2D(filters = 128, kernel_size=(3,3), activation='relu', padding = 'same'))

model.add(keras.layers.Dropout(rate=0.25))

model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(keras.layers.Conv2D(filters = 256, kernel_size=(3,3), activation='relu', padding = 'same'))

model.add(keras.layers.Dropout(rate=0.25))

model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(keras.layers.Flatten())

model.add(keras.layers.Dropout(rate=0.25))

model.add(keras.layers.Dense(units = 512, activation = 'relu'))

model.add(keras.layers.Dropout(rate=0.25))

model.add(keras.layers.Dense(units = 128, activation = 'relu'))

model.add(keras.layers.Dropout(rate=0.25))

model.add(keras.layers.Dense(units = 1, activation = 'sigmoid'))

# load weights
#model.load_weights("weights.best.hdf5")

model.compile(optimizer = 'adam',loss = 'binary_crossentropy', metrics = ['accuracy'])

### Using tensorboard callbacks
tbCallBack = keras.callbacks.TensorBoard(log_dir='./Model2/Drop/TBGraph', histogram_freq=0, write_graph=True, write_images=True)

# checkpoint Callback
filepath="./Model2/Drop/checkpoint/weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

time_callback = TimeHistory()

callbacks_list = [checkpoint, tbCallBack, time_callback]

model.fit_generator(train_generator, steps_per_epoch=689, epochs = 50,validation_data=validation_generator, callbacks=callbacks_list)

model2_drop_time = sum(time_callback.times)



# MODEL 2 WITH BATCH NORMALIZATION
model = keras.Sequential()

model.add(keras.layers.Conv2D(filters = 64, kernel_size=(3,3), input_shape = (64,64,3), activation='relu', padding = 'same'))

model.add(keras.layers.BatchNormalization())

model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(keras.layers.Conv2D(filters = 128, kernel_size=(3,3), activation='relu', padding = 'same'))

model.add(keras.layers.BatchNormalization())

model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(keras.layers.Conv2D(filters = 256, kernel_size=(3,3), activation='relu', padding = 'same'))

model.add(keras.layers.BatchNormalization())

model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(units = 512, activation = 'relu'))

model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Dense(units = 128, activation = 'relu'))

model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Dense(units = 1, activation = 'sigmoid'))

# load weights
# model.load_weights("weights.best.hdf5")

model.compile(optimizer = 'adam',loss = 'binary_crossentropy', metrics = ['accuracy'])

model.fit_generator(train_generator, steps_per_epoch=689, epochs = 50,validation_data=validation_generator)

### Using tensorboard callbacks
tbCallBack = keras.callbacks.TensorBoard(log_dir='./Model2/Norm/TBGraph', histogram_freq=0, write_graph=True, write_images=True)

# checkpoint Callback
filepath="./Model2/Norm/checkpoint/weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

time_callback = TimeHistory()

callbacks_list = [checkpoint, tbCallBack, time_callback]

model.fit_generator(train_generator, steps_per_epoch=689, epochs = 50,validation_data=validation_generator, callbacks=callbacks_list)

model2_norm_time = sum(time_callback.times)



# MODEL 2 WITH DROPOUT + BATCH NORMALIZATION
model = keras.Sequential()

model.add(keras.layers.Conv2D(filters = 64, kernel_size=(3,3), input_shape = (64,64,3), activation='relu', padding = 'same'))

model.add(keras.layers.BatchNormalization())

model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(keras.layers.Conv2D(filters = 128, kernel_size=(3,3), activation='relu', padding = 'same'))

model.add(keras.layers.BatchNormalization())

model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(keras.layers.Conv2D(filters = 256, kernel_size=(3,3), activation='relu', padding = 'same'))

model.add(keras.layers.BatchNormalization())

model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(keras.layers.Flatten())

model.add(keras.layers.Dropout(rate=0.25))

model.add(keras.layers.Dense(units = 512, activation = 'relu'))

model.add(keras.layers.Dropout(rate=0.25))

model.add(keras.layers.Dense(units = 128, activation = 'relu'))

model.add(keras.layers.Dropout(rate=0.25))

model.add(keras.layers.Dense(units = 10, activation = 'softmax'))

# load weights
model.load_weights("weights.best.hdf5")

model.compile(optimizer = 'adam',loss = 'categorical_crossentropy', metrics = ['accuracy'])

# specify model directory
model_dir = './CIFAR-10_Models/Model2/DropNorm2'
### Using tensorboard callbacks
tbCallBack = keras.callbacks.TensorBoard(log_dir=model_dir+'/TBGraph', histogram_freq=0, write_graph=True, write_images=True)

# checkpoint Callback
checkpoint = ModelCheckpoint(model_dir+"/checkpoint/weights.best.hdf5",
                             monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

time_callback = TimeHistory()

callbacks_list = [checkpoint, tbCallBack, time_callback]

model.fit_generator(train_generator, steps_per_epoch=train_steps, epochs = 25, verbose = 2,
                    validation_data=validation_generator, callbacks=callbacks_list)

model2_dropnorm_time = sum(time_callback.times)


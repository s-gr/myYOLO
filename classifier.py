import os
import random
import time

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from keras import applications
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, image

# Setting parameters
dir_train = 'dogscats/train'
dir_val = 'dogscats/valid'
batch_size = 16
epochs = 1

# Building the Model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Adding FC Layers
model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

if os.path.isfile('1st_model.h5'):
    model.load_weights('1st_model.h5')

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# rescale images
train_data_gen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_data_gen = ImageDataGenerator(rescale=1. / 255)

train_set = train_data_gen.flow_from_directory(
    dir_train,
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='binary'
)
test_set = test_data_gen.flow_from_directory(
    dir_val,
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='binary'
)

print(model.summary())

# tic
tic = time.time()

history = model.fit_generator(
    train_set,
    steps_per_epoch=2000 // batch_size,
    epochs=epochs,
    validation_data=test_set,
    validation_steps=800 // batch_size
)

# toc
toc = time.time()

# Computation time is
print('Computation Time is: ' + str(int((toc - tic) // pow(60, 2))).zfill(2) + ':'
      + str(int(((toc - tic) % pow(60, 2)) // 60)).zfill(2) + ':'
      + str(int(((toc - tic) % pow(60, 2)) % 60)).zfill(2)
      )

model.save_weights('1st_model.h5')

image_path = './dogscats/test1/' + random.choice(os.listdir('./dogscats/test1/'))
plt.figure()
plt.imshow(mpimg.imread(image_path))
plt.show()

test_image = image.load_img(image_path, target_size=(150, 150))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = model.predict(test_image)

if result >= 0.5:
    prediction = 'dog'
else:
    prediction = 'cat'

print(prediction)

print(history.history.keys())
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.ylim(0, 1)
plt.grid(axis='both')
plt.show()

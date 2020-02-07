"""
Source: https://www.tensorflow.org/tutorials/images/classification
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import time

import matplotlib.pyplot as plot
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator

# Declaration of directory constants
PATH = os.path.join(os.path.dirname(__file__), 'dataset')
TRAIN_DIR = os.path.join(PATH, 'training')
TEST_DIR = os.path.join(PATH, 'testing')

# Enter model .h5 filename to continue training
# Latest best model is 1581042441.h5 (15 epochs, Dropout == 0.1)
'''DO NOT TRAIN FURTHER -- MODEL *WILL* OVERFIT'''
MODEL_NAME = '1581042441.h5'
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', MODEL_NAME)

# Training constants and hyperparameters
BATCH_SIZE = 20
EPOCHS = 15

# Dimensions based on smallest dimensions from images in dataset
IMAGE_WIDTH = 173
IMAGE_HEIGHT = 167

# Fetches data from data set directories
test_img_generator = ImageDataGenerator(rescale=1.0 / 255)
train_img_generator = ImageDataGenerator(rescale=1.0 / 255)

train_data_gen = train_img_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                         directory=TRAIN_DIR,
                                                         shuffle=True,
                                                         target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                                                         class_mode='categorical')

test_data_gen = test_img_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                       directory=TEST_DIR,
                                                       shuffle=True,
                                                       target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                                                       class_mode='categorical')

# Model architecture
if MODEL_NAME:
    model = load_model(MODEL_PATH)      # Loads existing model
    print('Model has been loaded')
else:
    print('No model was loaded. Creating a new one.')
    model = Sequential(name='PokemonClassificationModel', layers=[
        Conv2D(filters=32,
               kernel_size=(3, 3),
               padding='same',
               activation='relu',
               input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.2),
        Conv2D(filters=64,
               kernel_size=(3, 3),
               padding='same',
               activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.2),
        Conv2D(filters=64,
               kernel_size=(3, 3),
               padding='same',
               activation='relu'),
        Flatten(),
        Dense(units=320,
              activation='relu'),
        Dense(units=4,
              activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

# Initiates training
history = model.fit_generator(generator=train_data_gen,
                              steps_per_epoch=BATCH_SIZE,
                              epochs=EPOCHS,
                              validation_data=test_data_gen,
                              validation_steps=BATCH_SIZE)

# Initiates variables for performance metrics
accuracy = history.history['accuracy']
test_accuracy = history.history['val_accuracy']

loss = history.history['loss']
test_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

# Plots accuracy metrics
plot.figure(figsize=(8, 8))
plot.subplot(1, 2, 1)
plot.plot(epochs_range, accuracy, label='Training Accuracy')
plot.plot(epochs_range, test_accuracy, label='Testing Accuracy')
plot.legend(loc='lower right')
plot.title('Training and Validation Accuracy')

# Plots loss metrics
plot.subplot(1, 2, 2)
plot.plot(epochs_range, loss, label='Training Loss')
plot.plot(epochs_range, test_loss, label='Testing Loss')
plot.legend(loc='upper right')
plot.title('Training and Validation Loss')
plot.show()

# Evaluates model
loss, accuracy = model.evaluate_generator(generator=test_data_gen,
                                          verbose=2)
print('Model accuracy: {:5.2f}%'.format(100 * accuracy))

# Saves model into HD5F file
t = time.time()
export_path = 'models/{}.h5'.format(int(t))
model.save(filepath=export_path, overwrite=False)
print('Model has been saved!')

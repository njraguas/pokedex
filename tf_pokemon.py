"""
Source: https://www.tensorflow.org/tutorials/images/cnn
Problem:
    1. Too overfit -- Fix preparation of training dataset
"""

import os
import pathlib

import matplotlib.pyplot as plot
import numpy as np
import tensorflow as tf
from keras import layers, models

AUTOTUNE = tf.data.experimental.AUTOTUNE

IMAGE_WIDTH = 173
IMAGE_HEIGHT = 167
BATCH_SIZE = 10

TRAIN_DIR = pathlib.Path('dataset/training')
TEST_DIR = pathlib.Path('dataset/testing')

list_train_set = tf.data.Dataset.list_files(str(TRAIN_DIR/'*/*'))
list_test_set = tf.data.Dataset.list_files(str(TEST_DIR/'*/*'))

train_count = len(list(TRAIN_DIR.glob('*/*.jpg')))
test_count = len(list(TEST_DIR.glob('*/*.jpg')))

TRAIN_STEPS_PER_EPOCH = np.ceil(train_count/BATCH_SIZE).astype(int)

CLASS_NAMES = np.array([item.name for item in TRAIN_DIR.glob('*')])


def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    return parts[-2] == CLASS_NAMES


def decode_image(this_image):
    this_image = tf.image.decode_jpeg(this_image, channels=3)
    this_image = tf.image.convert_image_dtype(this_image, tf.float32)

    return tf.image.resize(this_image, [IMAGE_WIDTH, IMAGE_HEIGHT])


def process_path(file_path):
    this_label = get_label(file_path)

    this_image = tf.io.read_file(file_path)
    this_image = decode_image(this_image)

    return this_image, this_label


def prepare_for_training(dataset, cache=True, shuffle_buffer_size=1000):
    if cache:
        if isinstance(cache, str):
            dataset = dataset.cache(cache)
        else:
            dataset = dataset.cache()

    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
    #dataset = dataset.repeat()
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)

    return dataset


def show_batch(image_batch, label_batch):
    plot.figure(figsize=(10, 10))

    for n in range(BATCH_SIZE):
        axis = plot.subplot(5, 5, n+1)

        plot.imshow(image_batch[n])
        plot.title(CLASS_NAMES[label_batch[n] == 1][0].title())
        plot.axis('off')
        plot.waitforbuttonpress()


labeled_train_set = list_train_set.map(process_path, num_parallel_calls=AUTOTUNE)
labeled_test_set = list_test_set.map(process_path, num_parallel_calls=AUTOTUNE)

train_dataset = prepare_for_training(labeled_train_set)
test_dataset = prepare_for_training(labeled_test_set)

image_batch, label_batch = next(iter(train_dataset))
test_image_batch, test_label_batch = next(iter(test_dataset))

# show_batch(image_batch.numpy(), label_batch.numpy())

model = models.Sequential()
model.add(layers.Conv2D(filters=32,
                        kernel_size=(3, 3),
                        activation='relu',
                        input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(filters=64,
                        kernel_size=(3, 3),
                        activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(filters=64,
                        kernel_size=(3, 3),
                        activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(units=128,
                       activation='relu'))
model.add(layers.Dense(units=4,
                       activation='softmax'))
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x=image_batch,
                    y=label_batch,
                    epochs=10,
                    validation_data=(test_image_batch, test_label_batch),
                    steps_per_epoch=TRAIN_STEPS_PER_EPOCH,
                    validation_steps=BATCH_SIZE)

plot.plot(history.history['accuracy'],
          label='accuracy')
plot.plot(history.history['val_accuracy'],
          label='val_accuracy')
plot.xlabel('Epoch')
plot.ylabel('Accuracy')
plot.ylim([0.5, 1])
plot.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_image_batch, test_label_batch, verbose=2, steps=TRAIN_STEPS_PER_EPOCH)
print(test_acc)

plot.show()
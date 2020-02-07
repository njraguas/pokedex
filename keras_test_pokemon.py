"""
Sources:
    https://www.tensorflow.org/tutorials/iamges/transfer_lerning_with_hub
    https://www.tensorflow.org/tutorials/keras/save_and_load
    https://www.tensorflow.org/guide/keras/save_and_serialize
"""

from keras import models
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plot
from keras.utils import get_file
from PIL import Image

import os
import pathlib
import numpy as np
import cv2

# Declaration of directory and model constants
MODEL_NAME = '1580977596.h5'
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', MODEL_NAME)

PATH = os.path.join(os.path.dirname(__file__), 'dataset')
TEST_DIR = os.path.join(PATH, 'testing')

GLOB_TEST_DIR = pathlib.Path('dataset/testing')
CLASS_NAMES = np.array([item.name for item in GLOB_TEST_DIR.glob('*')])

batch_size = 20

# Based on smallest dimensions in entire data set
IMAGE_WIDTH = 173
IMAGE_HEIGHT = 167

# Loads model from MODEL_PATH
test_model = models.load_model(MODEL_PATH)

# Creates a Test Data Generator from TEST_DIR
test_img_generator = ImageDataGenerator(rescale=1.0 / 255)
test_data_gen = test_img_generator.flow_from_directory(batch_size=batch_size,
                                                       directory=TEST_DIR,
                                                       shuffle=True,
                                                       target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                                                       class_mode='categorical')

# Loads labels/classes from Test Data Generator
class_names = sorted(test_data_gen.class_indices.items(), key=lambda pair:pair[1])
class_names = np.array([key.title() for key, value in class_names])

# Evaluates loaded model and prints overall Test Accuracy
test_loss, test_accuracy = test_model.evaluate_generator(generator=test_data_gen)
print(test_accuracy)

# Loads single image from a URL and pre-processes it
test_image = get_file('pokemon-charmander.jpg', 'https://static1.gamerantimages.com/wordpress/wp-content/uploads/2019/11/pokemon-charmander.jpg')
test_image = Image.open(test_image).resize((IMAGE_WIDTH, IMAGE_HEIGHT))
test_image = np.array(test_image) / 255.0

# Converts image to 3 channels if image has 4 channels
if len(test_image.shape) > 2 and test_image.shape[2] == 4:
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGRA2BGR)

# Loads image to model and model creates prediction
result = test_model.predict(test_image[np.newaxis])
predicted_class = np.argmax(result[0], axis=-1)
confidence_level = np.amax(result[0]) * 100
predicted_label = class_names[predicted_class]

# Displays image and model's prediction
plot.imshow(test_image)
plot.axis('off')
_ = plot.title('{}: {:4f}%'.format(predicted_label.title(), confidence_level))
plot.show()

'''
# Loads a directory into the model
class_names = sorted(test_data_gen.class_indices.items(), key=lambda pair:pair[1])
class_names = np.array([key.title() for key, value in class_names])


# Creates prediction for each Image Batch from the Test Data Generator
for image_batch, label_batch in test_data_gen:
    predicted_batch = test_model.predict(image_batch)
    predicted_id = np.argmax(predicted_batch, axis=-1)
    predicted_label_batch = class_names[predicted_id]

    label_id = np.argmax(label_batch, axis=-1)

    # Displays the images from Image Batch and the prediction of the model
    plot.figure(figsize=(10, 9))
    plot.subplots_adjust(hspace=0.5)

    for n in range(20):
        plot.subplot(6, 5, n+1)
        plot.imshow(image_batch[n])
        plot.title(predicted_label_batch[n].title())
        plot.axis('off')

    plot.show()
'''

#!/usr/bin/env python

# Based on
# https://www.nas.nasa.gov/hecc/support/kb/multiple-cpu-nodes-and-training-in-tensorflow_644.html
# edited to use small obtainable data, picking up the code from
# https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C1/W4/ungraded_labs/C1_W4_Lab_1_image_generator_no_validation.ipynb
# wget https://storage.googleapis.com/tensorflow-1-public/course2/week3/horse-or-human.zip https://storage.googleapis.com/tensorflow-1-public/course2/week3/validation-horse-or-human.zip

import os
import json
import sys
import time
import pathlib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, AveragePooling2D, \
MaxPool2D, Flatten, Dense, BatchNormalization, Dropout, Activation

# https://www.intel.com/content/www/us/en/developer/articles/guide/guide-to-tensorflow-runtime-optimizations-for-cpu.html
tf.config.threading.set_inter_op_parallelism_threads(num_threads=8)
tf.config.threading.set_intra_op_parallelism_threads(num_threads=8)
tf.config.set_soft_device_placement(True)

#set seeds for testing
tf.random.set_seed(22)
base_dir = os.path.dirname(os.path.abspath(__file__))
# --------------- Set up TF_CONFIG --------------- #
# load the index and node info from the command line
index = int(sys.argv[1])
verbose = index < 1
nodes=[]
# node names for each node, append a port # to the names
for i in range(2,len(sys.argv)):
    nodes.append(sys.argv[i] + ':2001')
# set TF_CONFIG variable that MultiWorkerMirroredStrategy needs
os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': nodes
    },
    'task': {'type': 'worker', 'index': index}
})
print(os.environ['TF_CONFIG'])
'''
verbose = 1
index = 0
'''
#if(verbose):
    #tf.debugging.set_log_device_placement(True)
#set up strategies
#strategy = tf.distribute.OneDeviceStrategy(device='/device:CPU:0')
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy \
(tf.distribute.experimental.CollectiveCommunication.AUTO)
#strategy = tf.distribute.MirroredStrategy()
EPOCHS = 5
#batch size per worker on each shard of the dataset
BATCH = 128
LEARNING_RATE = 0.001
IMG_WIDTH, IMG_HEIGHT = 150, 150
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_dir = base_dir + '/train'
train_dir = pathlib.Path(train_dir)
image_count = len(list(train_dir.glob('*/*.png')))
CLASS_NAMES = np.array([item.name for item in train_dir.glob('*') \
if item.name != "LICENSE.txt"])
def get_label(file_path):
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, '/')
    # The second to last is the class-directory
    return parts[-2] == 'human'
def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_png(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])
def process_path(file_path):
    label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label
def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000, \
num_workers=1, index=0):
    # This is a small dataset, only load it once, and 
    # keep it in memory.
    # use `.cache(filename)` to cache preprocessing work for 
    # datasets that don't fit in memory.
    ds.shard(num_workers, index)
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()
    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    # Repeat forever
    ds = ds.repeat()
    ds = ds.batch(BATCH)
    # `prefetch` lets the dataset fetch batches in the background while 
    # the model is training.
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

#list_ds = tf.data.Dataset.list_files(str(train_dir/'*/*'))
#labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
#train_ds = prepare_for_training(labeled_ds, num_workers=len(nodes), \
# index=index)
# imagegen = ImageDataGenerator(rescale=1./255)
# train = imagegen.flow_from_directory(directory='./data/train', \
# batch_size=BATCH, target_size=(224,224))
# val = imagegen.flow_from_directory(directory='./val', batch_size=BATCH, \
#target_size=(224,224))

def build_vgg_a():

    model = tf.keras.models.Sequential([
        # Note the input shape is the desired size of the image 150x150 with 3 bytes color
        # This is the first convolution
        tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The second convolution
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        # The third convolution
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        # # The fourth convolution (You can uncomment the 4th and 5th conv layers later to see the effect)
        # tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        # tf.keras.layers.MaxPooling2D(2,2),
        # # The fifth convolution
        # tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        # tf.keras.layers.MaxPooling2D(2,2),
        # Flatten the results to feed into a DNN
        tf.keras.layers.Flatten(),
        # 512 neuron hidden layer
        tf.keras.layers.Dense(512, activation='relu'),
        # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('horses') and 1 for the other ('humans')
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    global verbose
    if verbose:
        model.summary()
    from tensorflow.keras.optimizers import RMSprop
    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(learning_rate=0.001),
                  metrics=['accuracy'])
    return model

#use the distributed strategy
with strategy.scope():
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    ### # All images will be rescaled by 1./255
    ### train_datagen = ImageDataGenerator(rescale=1/255)
    ### validation_datagen = ImageDataGenerator(rescale=1/255)
    ###
    ### # Flow training images in batches of 128 using train_datagen generator
    ### train_generator = train_datagen.flow_from_directory(
    ###         './train/',  # This is the source directory for training images
    ###         target_size=(150, 150),  # All images will be resized to 150x150
    ###         batch_size=128,
    ###         # Since you used binary_crossentropy loss, you need binary labels
    ###         class_mode='binary')
    ###
    ### # Flow training images in batches of 128 using train_datagen generator
    ### validation_generator = validation_datagen.flow_from_directory(
    ###         './val/',  # This is the source directory for training images
    ###         target_size=(150, 150),  # All images will be resized to 150x150
    ###         batch_size=32,
    ###         # Since you used binary_crossentropy loss, you need binary labels
    ###         class_mode='binary')


    list_ds = tf.data.Dataset.list_files(str(train_dir/'*/*'))
    labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    print(strategy.num_replicas_in_sync)
    train_ds = prepare_for_training(labeled_ds, \
        num_workers=strategy.num_replicas_in_sync, index=index)
    model = build_vgg_a()

start = time.time()
model.fit(train_ds,
    epochs=EPOCHS,
    steps_per_epoch=int(image_count/BATCH),
    verbose=verbose
    )
if(verbose):
  print('took ' + str((time.time()-start)/60) + ' minutes')

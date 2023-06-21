import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from zipfile import ZipFile
from PIL import Image
import glob
import time

start_time = time.time()
seed = 42

tf.random.set_seed(seed)

import tensorflow as tf
import os
import shutil

# set the directories for the image files
#file_path = "crops/train_image_crops/*.jpg"
#file_mask_path = "crops/train_mask_crops/*.jpg"

file_path = "membrane/train/image/*.png"
file_mask_path = "membrane/train/label/*.png"

image_file = sorted(glob.glob(file_path))
mask_file = sorted(glob.glob(file_mask_path))

img_dir = image_file
seg_dir = mask_file

# create a dataset from the file paths and labels

dataset = tf.data.Dataset.from_tensor_slices((img_dir, seg_dir))


# define e a function to load and preprocess the images and segmentation images
def load_image(file_path, seg_path):
    # load the image file
    print("1")
    img = tf.io.read_file(file_path)
    # decode the image file
    img = tf.image.decode_jpeg(img, channels=3)
    # preprocess the image (e.g. resize, normalize, etc.)
    img = tf.image.resize(img, [256, 256])
    img = tf.cast(img, tf.float32) / 255.0
    
    # load the segmentation image file
    seg = tf.io.read_file(seg_path)
    # decode the segmentation image file
    seg = tf.squeeze(tf.io.decode_gif(seg), axis=0)
    # preprocess the segmentation image (e.g. resize, normalize, etc.)
    seg = tf.image.resize(seg, [256, 256])
    seg = tf.cast(seg, tf.float32) / 255.0
    return img, seg

# map the load_image function to the dataset


dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

# define the batch size and shuffle the dataset
batch_size = 64

# define the size of the validation set as a fraction of the total dataset size
val_size = 0.2

# calculate the number of validation examples
val_size = int(len(img_dir) * val_size)

# shuffle the dataset
dataset = dataset.shuffle(buffer_size=1000)

# split the dataset into training and validation datasets
val_dataset = dataset.take(val_size)
train_dataset = dataset.skip(val_size)

class Augment(keras.layers.Layer):
    def __init__(self, seed=42):
        super().__init__()
        self.augment_inputs = keras.layers.RandomFlip(mode="horizontal", seed=seed)
        self.augment_labels = keras.layers.RandomFlip(mode="horizontal", seed=seed)
        
    def call(self, inputs, labels):
        inputs = self.augment_inputs(inputs)
        labels = self.augment_labels(labels)
        return inputs, labels
    
    
    
train_batches = (
    train_dataset
    .shuffle(1000)
    .batch(batch_size)
    .map(Augment())
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

val_batches = val_dataset.batch(batch_size)


def display(display_list):
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.utils.array_to_img(display_list[i]), cmap="gray")
    plt.axis('off')
  plt.show()


for images, masks in train_batches.take(2):
  sample_image, sample_mask = images[0], masks[0]
  display([sample_image, sample_mask])    
  
  
def encoder_block(layer_in, n_filters, batchnorm=True):
    # weight initialization
    init = tf.random_normal_initializer(0.02, seed=seed)
    # add downsampling layer
    g = layers.Conv2D(n_filters, (4, 4), strides=(2, 2), padding='same',
                     kernel_initializer=init)(layer_in)
    # conditionally add batch normalization
    if batchnorm:
        g = layers.BatchNormalization()(g, training=True)
    # leaky relu activation
    g = layers.LeakyReLU(alpha=0.2)(g)
    return g

# define a decoder block
def decoder_block(layer_in, skip_in, n_filters, dropout=True):
    # weight initialization
    init = tf.random_normal_initializer(stddev=0.02, seed=seed)
    # add upsampling layer
    g = layers.Conv2DTranspose(n_filters, (4, 4), strides=(2, 2), padding='same',
                              kernel_initializer=init)(layer_in)
    # add batch normalization
    g = layers.BatchNormalization()(g, training=True)
    # conditionally add dropout
    if dropout:
        g = layers.Dropout(0.3)(g, training=True)
    # merge with skip connection
    g = layers.Concatenate()([g, skip_in])
    g = layers.Activation('relu')(g)
    return g

# define the U-Net model
def UNet(image_shape=(256, 256, 3)):
    # weight initialization
    init = tf.random_normal_initializer(stddev=0.02, seed=seed)
    # image input
    in_image = keras.Input(shape=image_shape)
    
    # encoder model: C64-C128-C256-C512-C512-C512-C512-C512
    e1 = encoder_block(in_image, 64, batchnorm=False)
    e2 = encoder_block(e1, 128)
    e3 = encoder_block(e2, 256)
    e4 = encoder_block(e3, 512)
    e5 = encoder_block(e4, 512)
    e6 = encoder_block(e5, 512)
    e7 = encoder_block(e6, 512)
    
    # bottleneck, no batch norm and relu
    b = layers.Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(e7)
    b = layers.Activation('relu')(b)
    
    # decoder model: CD512-CD1024-CD1024-C1024-C1024-C512-C256-C128
    d1 = decoder_block(b, e7, 512)
    d2 = decoder_block(d1, e6, 512)
    d3 = decoder_block(d2, e5, 512)
    d4 = decoder_block(d3, e4, 512, dropout=False)
    d5 = decoder_block(d4, e3, 256, dropout=False)
    d6 = decoder_block(d5, e2, 128, dropout=False)
    d7 = decoder_block(d6, e1, 64, dropout=False)
    
    # output 
    g = layers.Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d7)
    out_image = layers.Activation('sigmoid')(g)
    model = keras.Model(in_image, out_image)
    return model

image_shape = (256, 256, 3)
model = UNet(image_shape)
model.summary()


def iou_score(y_true, y_pred, smooth=1):
    
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])

  
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)

    
    iou = (intersection + smooth) / (union + smooth)

    return iou

def calculate_accuracy(y_true, y_pred):
    y_pred = tf.round(y_pred)  
    correct_pixels = tf.reduce_sum(tf.cast(tf.equal(y_true, y_pred), dtype=tf.float32))
    total_pixels = tf.cast(tf.size(y_true), dtype=tf.float32)
    accuracy = correct_pixels / total_pixels
    return accuracy

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=[calculate_accuracy, iou_score])
history = model.fit(train_batches, epochs=10, validation_data=val_batches)



def show_predictions(dataset=val_batches):
  if dataset:
    for image, mask in dataset.take(5):
      pred_mask = model.predict(mask)
      display([image[0], mask[0], pred_mask[0]])
      
show_predictions()


model.save("bio-unet.h5")      

end_time = time.time()

time = end_time-start_time
print("Trainin time: ", time)

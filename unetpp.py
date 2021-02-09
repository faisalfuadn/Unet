# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 18:48:20 2021

@author: suantara
"""


def unetpp (img_size, n_label, nb_filters=32, depth=4, act='relu'):
  from keras import layers
  from keras import models
  from . import conv2d
  """
  img_size= 2D tuple image size
  n_labels= int, no of labels
  act= activation function
  nb_filters=int, number of filter
  depth= int, depth of unet layer
  """
  inputs= layers.Input((img_size)+(3,))
  skip_connection=[]

  #contractive
  x= inputs
  for i in range (depth):
    x = conv2d(x, nb_filters, act=act)
    skip_connection.append(x)
    x = layers.MaxPooling2D((2, 2))(x)
    nb_filters= nb_filters * 2
  
  x= conv2d(x, nb_filters, act=act)

  #Dense Module connection
  up3_1= layers.UpSampling2D(size=(2,2))(skip_connection[3])
  dense3_1= conv2d(layers.concatenate([skip_connection[2], up3_1]), 128, act=act)

  up2_1= layers.UpSampling2D(size=(2,2))(skip_connection[2])
  dense2_1= conv2d(layers.concatenate([skip_connection[1], up2_1]), 64, act=act)
  up2_2= layers.UpSampling2D(size=(2,2))(dense3_1)
  dense2_2= conv2d(layers.concatenate([dense2_1, skip_connection[1], up2_2]), 64, act=act)

  up1_1= layers.UpSampling2D(size=(2,2))(skip_connection[1])
  dense1_1= conv2d(layers.concatenate([up1_1, skip_connection[0]]), 32, act=act)
  up1_2= layers.UpSampling2D(size=(2,2))(up2_1)
  dense1_2= conv2d(layers.concatenate([dense1_1, skip_connection[0], up1_2]), 32, act=act)
  up1_3= layers.UpSampling2D(size=(2,2))(up2_2)
  dense1_3= conv2d(layers.concatenate([dense1_1, dense1_2, skip_connection[0], up1_3]), 32, act=act)

  dense_module=[dense1_3, dense2_2, dense3_1]

  #expansive part
  for i in reversed(range(depth)):
    nb_filters= nb_filters // 2
    x = layers.UpSampling2D(size=(2, 2))(x)
    if i >= 3:
      x = layers.concatenate([skip_connection[i], x], axis=-1)
    elif i < 3:
      x = layers.concatenate([skip_connection[i], x, dense_module[i]], axis=-1)
    x = conv2d(x, nb_filters, act=act)
  
  final_conv = layers.Conv2D(n_label, (1, 1), padding='same')(x)
  final_conv = layers.core.Activation('sigmoid')(final_conv)
  model = models.Model(inputs=inputs, outputs=final_conv)
  return model
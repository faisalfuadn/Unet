# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 15:43:35 2021

@author: suantara
"""


def unet(img_size, n_labels, act='relu', nb_filters=32, depth=4):
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

  #expansive part
  for i in reversed(range(depth)):
    nb_filters= nb_filters // 2
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.concatenate([skip_connection[i], x], axis=-1)
    x = conv2d(x, nb_filters, act=act)
  
  final_conv = layers.Conv2D(n_labels, (1, 1), padding='same')(x)
  final_conv = layers.core.Activation('sigmoid')(final_conv)
  model = models.Model(inputs=inputs, outputs=final_conv)
  return model
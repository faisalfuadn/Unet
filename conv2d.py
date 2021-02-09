# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 10:07:23 2021

@author: suantara
"""

def conv2d(input_tensor, nb_filter, kernel_size=3, act='relu'):
  from keras import layers
  """
  input tensor= 3D tensor, H X W x C
  nb_filter= no of filter
  kernel_size= filter size
  act= activation function
  """
  x = layers.Conv2D(nb_filter, kernel_size, padding='same')(input_tensor)
  x = layers.BatchNormalization(axis=-1)(x)
  x = layers.Activation(act)(x)
  x = layers.Conv2D(nb_filter, kernel_size, padding='same')(x)
  x = layers.BatchNormalization(axis=-1)(x)
  x = layers.Activation(act)(x)
  return x
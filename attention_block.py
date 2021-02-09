# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 18:45:29 2021

@author: suantara
"""


def attention_block(skip_tensor, down_tensor, nb_filters, act='relu'):
  from keras import layers
  import keras.backend as K
  """
  x= 3D tensor from skip connection
  g= 3D tensor from down layer
  nb_filter= int, number of filter
  act= activation function
  """
  #get shape of every tensor
  skip_tensor_shape= K.int_shape(skip_tensor)
  down_tensor_shape= K.int_shape(down_tensor)

  #equalize the tensor shape
  conv_down= layers.Conv2D(filters=nb_filters, kernel_size= 1, strides= 1, padding='same')(down_tensor)
  strided_conv_skip= layers.Conv2D(filters=nb_filters, 
                         kernel_size= 1, 
                         strides= (skip_tensor_shape[1]//down_tensor_shape[1],
                                   skip_tensor_shape[2]//down_tensor_shape[2]),
                         padding='same')(skip_tensor)
  
  #summation of the two tensor
  summation_tensor= layers.add([conv_down, strided_conv_skip])
  summation_tensor= layers.BatchNormalization(axis=-1)(summation_tensor)
  summation_tensor= layers.core.Activation(act)(summation_tensor)

  #1x1x1 conv 
  conv1x1= layers.Conv2D(filters=1, kernel_size=1, padding='same')(summation_tensor)
  conv1x1= layers.BatchNormalization(axis=-1)(conv1x1)
  conv1x1= layers.core.Activation('sigmoid')(conv1x1)

  #Up-sampling, to match skip connection shape
  up_conv1x1x1= layers.UpSampling2D(size=(skip_tensor_shape[1] // K.int_shape(conv1x1)[1],
                                          skip_tensor_shape[2] // K.int_shape(conv1x1)[2]))(conv1x1)                                    
  #scaler resampler, from 1x1x1 to 1x1x3
  up_conv1x1x1= layers.Lambda(lambda x, repeat: K.repeat_elements(x, repeat, axis=3),
                              arguments={'repeat': skip_tensor_shape[3]})(up_conv1x1x1)

  #multiplication from scaler and skip connection
  multiplication= layers.multiply([up_conv1x1x1, skip_tensor])
  final_conv= layers.Conv2D(filters=skip_tensor_shape[-1], kernel_size=1, padding='same')(multiplication)
  return final_conv
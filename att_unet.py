# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 18:45:58 2021

@author: suantara
"""


def att_unet(img_size, n_label, nb_filters=32, depth=4, act='relu'):
  from keras import layers
  from keras import models
  from . import attention_block
  from . import conv2d
  """
  img_size= 2D tuple image size
  n_labels= int, no of labels
  nb_filters= int, number of filters
  depth=int, layer depth
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
    att = attention_block(skip_connection[i], x, nb_filters, act=act)
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.concatenate([att, x], axis=-1)
    x = conv2d(x, nb_filters, act=act)
  
  final_layer = layers.Conv2D(n_label, (1, 1), padding='same')(x)
  final_layer = layers.core.Activation('sigmoid')(x)
  model = models.Model(inputs=inputs, outputs=final_layer)
  return model
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 18:44:04 2021

@author: suantara
"""


def residual_unet(img_size, n_labels, nb_filters= 32, depth= 4, act='relu', ratio=0.1):
  from keras import layers
  from keras import models
  import keras.backend as K
  from . import conv2d
  """
  img_size= 2D tuple image size
  n_labels= int, no of labels
  nb_filters= int, number of filter
  depth= int, layers depth
  act= activation function
  ratio= float 0 - 1 for residual scaler
  """
  inputs= layers.Input((img_size)+(3,))
  skip_connection=[]

  #contractive
  x= inputs
  for i in range (depth):
    residual = layers.Conv2D(nb_filters, 3, padding="same")(x)
    x = conv2d(x, nb_filters, act=act)
    x = layers.Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
                  output_shape=K.int_shape(x)[1:],
                  arguments={'scale': ratio})([x, residual])
    skip_connection.append(x)
    x = layers.MaxPooling2D((2, 2))(x)
    nb_filters= nb_filters * 2
  x= conv2d(x, nb_filters, act=act)

  #expansive part
  for i in reversed(range(depth)):
    nb_filters= nb_filters // 2
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.concatenate([skip_connection[i], x], axis=-1)
    residual = layers.Conv2D(nb_filters, 1, padding="same")(x)
    x = conv2d(x, nb_filters, act=act)
    x = layers.Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
                      output_shape=K.int_shape(x)[1:],
                      arguments={'scale': ratio})([x, residual])
  
  final_conv = layers.Conv2D(n_labels, (1, 1), padding='same')(x)
  final_conv = layers.core.Activation('sigmoid')(final_conv)
  model = models.Model(inputs=inputs, outputs=final_conv)
  return model
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 19:03:32 2021

@author: suantara
"""


from keras import backend as K
def mish(inputs):
  return inputs * K.tanh(K.softplus(inputs))
from keras.utils.generic_utils import get_custom_objects
get_custom_objects().update({'mish': mish})
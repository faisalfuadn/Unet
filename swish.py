# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 19:03:00 2021

@author: suantara
"""


from keras.backend import sigmoid
def swish(x, beta = 1):
    return (x * sigmoid(beta * x))

from keras.utils.generic_utils import get_custom_objects
get_custom_objects().update({'swish': swish})
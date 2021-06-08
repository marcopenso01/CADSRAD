"""
@author: Marco Penso
"""
import tensorflow as tf
from keras.layers import *
import numpy as np
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.applications import ResNet50V2


def VGG16_model(images, nlabels, mode):
    
  from tensorflow.keras.applications import VGG16
  
  if mode == '2D':
    
  
  base_model = VGG16(input_shape = images.shape[1:],
                     include_top = False,
                     weights = 'imagenet')
  
  base_model.trainable = False
  
  x = layers.Flatten()(base_model.output)
  
  x = layers.Dense(512, init='normal', activation='relu')(x)
  
  x = layers.Dropout(0.5)(x)
  
  x = layers.Dense(512, init='normal', activation='relu')(x)
  
  x = layers.Dropout(0.5)(x)

  x = layers.Dense(nlabels, init='normal', activation='softmax')(x)

  model = tf.keras.models.Model(base_model.input, x, name='VGG16')
  
  return model
  
  
def InceptionV3_model(images, nlabels):
  
  base_model = InceptionV3(input_shape = images.shape[1:],
                           include_top = False,
                           weights = 'imagenet')
  
  base_model.trainable = False

  x = layers.Flatten()(base_model.output)

  x = layers.Dense(512, init='normal', activation='relu')(x)

  x = layers.Dropout(0.5)(x)

  x = layers.Dense(512, init='normal', activation='relu')(x)

  x = layers.Dropout(0.5)(x)

  x = layers.Dense(nlabels, init='normal', activation='softmax')(x)

  model = tf.keras.models.Model(base_model.input, x, name='InceptionV3')

  return model
  

def get_model(images, config):
    
  return config.model_handle(images, nlabels=config.nlabels, mode=config.data_mode)

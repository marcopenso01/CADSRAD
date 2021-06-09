"""
@author: Marco Penso
"""
import tensorflow as tf
from tensorflow.keras import layers 
from tensorflow.keras import Model 
from tensorflow.keras.layers import *

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.applications import ResNet50V2


def VGG16_model(input_tensor, nlabels, mode):
   
  if mode == '2D':
        
        if input_tensor.shape[-1] != 1:
            raise AssertionError('Inadequate input tensor shape. The input must have 1 channels')
        else:
            input_tensor_shape = Input(shape=(input_tensor.shape[1], input_tensor.shape[2], 1))
            images = Concatenate(axis=-1)([input_tensor_shape, input_tensor_shape, input_tensor_shape])
  
  if mode == '3D':
        
        if input_tensor.shape[-1] != 3:
            raise AssertionError('Inadequate input tensor shape. The input must have 3 channels')
        else:
            images = Input(
                 
  from tensorflow.keras.applications import VGG16    
  
  base_model = VGG16(input_tensor = images,
                     include_top = False,
                     weights = 'imagenet')
  
  base_model.trainable = False
  
  x = layers.Flatten()(base_model.output)
  
  x = layers.Dense(512, kernel_initializer=initializer, activation='relu')(x)
  
  x = layers.Dropout(0.5)(x)
  
  x = layers.Dense(512, kernel_initializer=initializer, activation='relu')(x)
  
  x = layers.Dropout(0.5)(x)

  output = layers.Dense(nlabels, kernel_initializer=initializer, activation='softmax')(x)

  model = Model(base_model.input, output, name='VGG16')
  
  return model
  

                
def InceptionV3_model(input_tensor, nlabels, mode):
   
  if mode == '2D':
        
        if input_tensor.shape[-1] != 1:
            raise AssertionError('Inadequate input tensor shape. The input must have 1 channels')
        else:
            input_tensor_shape = Input(shape=(input_tensor.shape[1], input_tensor.shape[2], 1))
            images = Concatenate(axis=-1)([input_tensor_shape, input_tensor_shape, input_tensor_shape])
  
  if mode == '3D':
        
        if input_tensor.shape[-1] != 3:
            raise AssertionError('Inadequate input tensor shape. The input must have 3 channels')
        else:
            images = Input(
                 
  from tensorflow.keras.applications import InceptionV3    
  
  base_model = InceptionV3(input_tensor = images,
                           include_top = False,
                           weights = 'imagenet')
  
  base_model.trainable = False
  
  x = layers.Flatten()(base_model.output)
  
  x = layers.Dense(512, kernel_initializer=initializer, activation='relu')(x)
  
  x = layers.Dropout(0.5)(x)
  
  x = layers.Dense(512, kernel_initializer=initializer, activation='relu')(x)
  
  x = layers.Dropout(0.5)(x)

  output = layers.Dense(nlabels, kernel_initializer=initializer, activation='softmax')(x)

  model = Model(base_model.input, output, name='InceptionV3')
  
  return model
                
  

def get_model(images, config):
    
  return config.model_handle(images, nlabels=config.nlabels, mode=config.data_mode)

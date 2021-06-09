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


def VGG16_model(input_tensor, config):
  
  mode = config.data_mode
  
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
         images = Input()
                 
  from tensorflow.keras.applications import VGG16
  
  base_model = VGG16(input_tensor = images,
                     include_top = False,
                     weights = 'imagenet')
  
  base_model.trainable = False
  
  x = layers.Flatten()(base_model.output)

  if len(config.dense_layer) != len(config.drop_rate):
      raise AssertionError('Inadequate model settings')
  
  else:
      
     for ii in range(len(config.dense_layer)):

         x = layers.Dense(config.dense_layer[ii], kernel_initializer=get_init(config.kernel_init), kernel_regularizer=get_reg(config.kernel_reg), activation='relu')(x)

         if config.drop_rate[ii] > 0 and config.drop_rate[ii] < 1:

            x = layers.Dropout(config.drop_rate[ii])(x)

  output = layers.Dense(config.nlabels, kernel_initializer=get_init(config.kernel_init), kernel_regularizer=get_reg(config.kernel_reg), activation='softmax')(x)

  model = Model(base_model.input, output, name='VGG16')
  
  return model
  

def InceptionV3_model(input_tensor, config):
  
  mode = config.data_mode
  
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
         images = Input()
                 
  from tensorflow.keras.applications import InceptionV3
  
  base_model = VGG16(input_tensor = images,
                     include_top = False,
                     weights = 'imagenet')
  
  base_model.trainable = False
  
  x = layers.Flatten()(base_model.output)

  if len(config.dense_layer) != len(config.drop_rate):
      raise AssertionError('Inadequate model settings')
  
  else:
      
     for ii in range(len(config.dense_layer)):

         x = layers.Dense(config.dense_layer[ii], kernel_initializer=get_init(config.kernel_init), kernel_regularizer=get_reg(config.kernel_reg), activation='relu')(x)

         if config.drop_rate[ii] > 0 and config.drop_rate[ii] < 1:

            x = layers.Dropout(config.drop_rate[ii])(x)

  output = layers.Dense(config.nlabels, kernel_initializer=get_init(config.kernel_init), kernel_regularizer=get_reg(config.kernel_reg), activation='softmax')(x)

  model = Model(base_model.input, output, name='InceptionV3')
  
  return model
                
 
def get_init(type='he_normal'):
               
  if type == 'he_normal':
      initial = tf.keras.initializers.HeNormal()
  if type == 'he_uniform':
      initial = tf.keras.initializers.HeUniform()
  if type == 'xavier_normal':
      initial = tf.keras.initializers.GlorotNormal()
  if type == 'xavier_uniform':
      initial = tf.keras.initializers.GlorotUniform()
  else:
      raise ValueError('Unknown initialisation requested: %s' % type)
  
  return initial
               

def get_reg(type='None'):
   
  if type == 'L1':
      regularize = tf.keras.regularizers.L1()
  if type == 'L2':
      regularize = tf.keras.regularizers.L2()
  if type == 'L1L2':
      regularize = tf.keras.regularizers.L1L2()
  if type == 'None':
      regularize = None
  else:
      raise ValueError('Unknown regularization requested: %s' % type)
  
  return regularize

               
def get_model(images, config):
  
  return config.model_handle(images, config)

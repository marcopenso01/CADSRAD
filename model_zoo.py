"""
@author: Marco Penso
"""
import tensorflow as tf
from tensorflow.keras import layers 
from tensorflow.keras import Model 
from tensorflow.keras.layers import *

from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.applications import ResNet50V2


def VGG16_model(input_tensor, nlabels, config):
  
  mode = config.data_mode
  
  if mode == '2D':
      if input_tensor.ndim > 3:
         raise AssertionError('Error tensor shape: expected input to have shape (N,x,y)')
      else:
         input_tensor_shape = Input(shape=(input_tensor.shape[1], input_tensor.shape[2], 1))
         images = Concatenate(axis=-1)([input_tensor_shape, input_tensor_shape, input_tensor_shape])
  
  if mode == '3D':    
      if input_tensor.ndim > 4:
         raise AssertionError('Error tensor shape: expected input to have shape (N,x,y,z)')
      elif input_tensor.shape[-1] != 3:
         raise AssertionError('Inadequate tensor shape. Input must have 3 channels (N,x,y,3)')
      else:
         images = Input(shape=(input_tensor.shape[1], input_tensor.shape[2], input_tensor.shape[3]))
  
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
   
  if nlabels > 2:
     output = layers.Dense(nlabels, kernel_initializer=get_init(config.kernel_init), kernel_regularizer=get_reg(config.kernel_reg), activation='softmax')(x)
  else:
     output = layers.Dense(nlabels-1, kernel_initializer=get_init(config.kernel_init), kernel_regularizer=get_reg(config.kernel_reg), activation='sigmoid')(x)

  model = Model(base_model.input, output, name='VGG16')
  
  return model
  

def InceptionV3_model(input_tensor, nlabels, config):
  
  mode = config.data_mode
  
  if mode == '2D':
      if input_tensor.ndim > 3:
         raise AssertionError('Error tensor shape: expected input to have shape (N,x,y)')
      else:
         input_tensor_shape = Input(shape=(input_tensor.shape[1], input_tensor.shape[2], 1))
         images = Concatenate(axis=-1)([input_tensor_shape, input_tensor_shape, input_tensor_shape])
  
  if mode == '3D':    
      if input_tensor.ndim > 4:
         raise AssertionError('Error tensor shape: expected input to have shape (N,x,y,z)')
      elif input_tensor.shape[-1] != 3:
         raise AssertionError('Inadequate tensor shape. Input must have 3 channels (N,x,y,3)')
      else:
         images = Input(shape=(input_tensor.shape[1], input_tensor.shape[2], input_tensor.shape[3]))
  
  base_model = InceptionV3(input_tensor = images,
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

  if nlabels > 2:
     output = layers.Dense(nlabels, kernel_initializer=get_init(config.kernel_init), kernel_regularizer=get_reg(config.kernel_reg), activation='softmax')(x)
  else:
     output = layers.Dense(nlabels-1, kernel_initializer=get_init(config.kernel_init), kernel_regularizer=get_reg(config.kernel_reg), activation='sigmoid')(x)

  model = Model(base_model.input, output, name='InceptionV3')
  
  return model
                

def ResNet50_model(input_tensor, nlabels, config):
  
  mode = config.data_mode
  
  if mode == '2D':
      if input_tensor.ndim > 3:
         raise AssertionError('Error tensor shape: expected input to have shape (N,x,y)')
      else:
         input_tensor_shape = Input(shape=(input_tensor.shape[1], input_tensor.shape[2], 1))
         images = Concatenate(axis=-1)([input_tensor_shape, input_tensor_shape, input_tensor_shape])
  
  if mode == '3D':    
      if input_tensor.ndim > 4:
         raise AssertionError('Error tensor shape: expected input to have shape (N,x,y,z)')
      elif input_tensor.shape[-1] != 3:
         raise AssertionError('Inadequate tensor shape. Input must have 3 channels (N,x,y,3)')
      else:
         images = Input(shape=(input_tensor.shape[1], input_tensor.shape[2], input_tensor.shape[3]))
  
  base_model = ResNet50(input_tensor = images,
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

  if nlabels > 2:
     output = layers.Dense(nlabels, kernel_initializer=get_init(config.kernel_init), kernel_regularizer=get_reg(config.kernel_reg), activation='softmax')(x)
  else:
     output = layers.Dense(nlabels-1, kernel_initializer=get_init(config.kernel_init), kernel_regularizer=get_reg(config.kernel_reg), activation='sigmoid')(x)

  model = Model(base_model.input, output, name='ResNet50')
  
  return model


def InceptionResNetV2_model(input_tensor, nlabels, config):
  
  mode = config.data_mode
  
  if mode == '2D':
      if input_tensor.ndim > 3:
         raise AssertionError('Error tensor shape: expected input to have shape (N,x,y)')
      else:
         input_tensor_shape = Input(shape=(input_tensor.shape[1], input_tensor.shape[2], 1))
         images = Concatenate(axis=-1)([input_tensor_shape, input_tensor_shape, input_tensor_shape])
  
  if mode == '3D':    
      if input_tensor.ndim > 4:
         raise AssertionError('Error tensor shape: expected input to have shape (N,x,y,z)')
      elif input_tensor.shape[-1] != 3:
         raise AssertionError('Inadequate tensor shape. Input must have 3 channels (N,x,y,3)')
      else:
         images = Input(shape=(input_tensor.shape[1], input_tensor.shape[2], input_tensor.shape[3]))
  
  base_model = InceptionResNetV2(input_tensor = images,
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

  if nlabels > 2:
     output = layers.Dense(nlabels, kernel_initializer=get_init(config.kernel_init), kernel_regularizer=get_reg(config.kernel_reg), activation='softmax')(x)
  else:
     output = layers.Dense(nlabels-1, kernel_initializer=get_init(config.kernel_init), kernel_regularizer=get_reg(config.kernel_reg), activation='sigmoid')(x)

  model = Model(base_model.input, output, name='InceptionResNetV2')
  
  return model


def EfficientNetB0_model(input_tensor, nlabels, config):
  
  mode = config.data_mode
  
  if mode == '2D':
      if input_tensor.ndim > 3:
         raise AssertionError('Error tensor shape: expected input to have shape (N,x,y)')
      else:
         input_tensor_shape = Input(shape=(input_tensor.shape[1], input_tensor.shape[2], 1))
         images = Concatenate(axis=-1)([input_tensor_shape, input_tensor_shape, input_tensor_shape])
  
  if mode == '3D':    
      if input_tensor.ndim > 4:
         raise AssertionError('Error tensor shape: expected input to have shape (N,x,y,z)')
      elif input_tensor.shape[-1] != 3:
         raise AssertionError('Inadequate tensor shape. Input must have 3 channels (N,x,y,3)')
      else:
         images = Input(shape=(input_tensor.shape[1], input_tensor.shape[2], input_tensor.shape[3]))
    
  base_model = EfficientNetB0(input_tensor = images,
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

  if nlabels > 2:
     output = layers.Dense(nlabels, kernel_initializer=get_init(config.kernel_init), kernel_regularizer=get_reg(config.kernel_reg), activation='softmax')(x)
  else:
     output = layers.Dense(nlabels-1, kernel_initializer=get_init(config.kernel_init), kernel_regularizer=get_reg(config.kernel_reg), activation='sigmoid')(x)

  model = Model(base_model.input, output, name='EfficientNetB0')
  
  return model


def EfficientNetB7_model(input_tensor, nlabels, config):
  
  mode = config.data_mode
  
  if mode == '2D':
      if input_tensor.ndim > 3:
         raise AssertionError('Error tensor shape: expected input to have shape (N,x,y)')
      else:
         input_tensor_shape = Input(shape=(input_tensor.shape[1], input_tensor.shape[2], 1))
         images = Concatenate(axis=-1)([input_tensor_shape, input_tensor_shape, input_tensor_shape])
  
  if mode == '3D':    
      if input_tensor.ndim > 4:
         raise AssertionError('Error tensor shape: expected input to have shape (N,x,y,z)')
      elif input_tensor.shape[-1] != 3:
         raise AssertionError('Inadequate tensor shape. Input must have 3 channels (N,x,y,3)')
      else:
         images = Input(shape=(input_tensor.shape[1], input_tensor.shape[2], input_tensor.shape[3]))
  
  base_model = EfficientNetB7(input_tensor = images,
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

  if nlabels > 2:
     output = layers.Dense(nlabels, kernel_initializer=get_init(config.kernel_init), kernel_regularizer=get_reg(config.kernel_reg), activation='softmax')(x)
  else:
     output = layers.Dense(nlabels-1, kernel_initializer=get_init(config.kernel_init), kernel_regularizer=get_reg(config.kernel_reg), activation='sigmoid')(x)

  model = Model(base_model.input, output, name='EfficientNetB7')
  
  return model


def ResNet50V2_model(input_tensor, nlabels, config):
  
  mode = config.data_mode
  
  if mode == '2D':
      if input_tensor.ndim > 3:
         raise AssertionError('Error tensor shape: expected input to have shape (N,x,y)')
      else:
         input_tensor_shape = Input(shape=(input_tensor.shape[1], input_tensor.shape[2], 1))
         images = Concatenate(axis=-1)([input_tensor_shape, input_tensor_shape, input_tensor_shape])
  
  if mode == '3D':    
      if input_tensor.ndim > 4:
         raise AssertionError('Error tensor shape: expected input to have shape (N,x,y,z)')
      elif input_tensor.shape[-1] != 3:
         raise AssertionError('Inadequate tensor shape. Input must have 3 channels (N,x,y,3)')
      else:
         images = Input(shape=(input_tensor.shape[1], input_tensor.shape[2], input_tensor.shape[3]))
  
  base_model = ResNet50V2(input_tensor = images,
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

  if nlabels > 2:
     output = layers.Dense(nlabels, kernel_initializer=get_init(config.kernel_init), kernel_regularizer=get_reg(config.kernel_reg), activation='softmax')(x)
  else:
     output = layers.Dense(nlabels-1, kernel_initializer=get_init(config.kernel_init), kernel_regularizer=get_reg(config.kernel_reg), activation='sigmoid')(x)

  model = Model(base_model.input, output, name='ResNet50V2')
  
  return model


def model_3d(input_tensor, nlabels):
    
  inputs = Input(shape=(input_tensor.shape[1], input_tensor.shape[2], input_tensor.shape[3], 1))
  print('input', inputs.shape)
  x = layers.Conv3D(filters=32, kernel_size=3, activation="relu", kernel_initializer='he_normal', padding='same')(inputs)
  print('con3d', x.shape)
  x = layers.MaxPool3D(pool_size=2)(x)
  print('max_pool', x.shape)
  x = layers.BatchNormalization()(x)
  print('batch', x.shape)

  x = layers.Conv3D(filters=64, kernel_size=3, activation="relu", kernel_initializer='he_normal', padding='same')(x)
  print('con3d', x.shape)
  x = layers.MaxPool3D(pool_size=2)(x)
  print('pool', x.shape)
  x = layers.BatchNormalization()(x)
  print('batch', x.shape)

  x = layers.Conv3D(filters=128, kernel_size=3, activation="relu", kernel_initializer='he_normal', padding='same')(x)
  print('conv3d',x.shape)
  x = layers.MaxPool3D(pool_size=2)(x)
  print('pool', x.shape)
  x = layers.BatchNormalization()(x)
  print('batch', x.shape)

  x = layers.GlobalAveragePooling3D()(x)
  print('globalPool', x.shape)
  x = layers.Dense(units=512, kernel_initializer='he_normal', activation="relu")(x)
  print('dense', x.shape)
  x = layers.Dropout(0.3)(x)
  print('drop', x.shape)
  
  if nlabels > 2:
    outputs = layers.Dense(units=nlabels, kernel_initializer='he_normal', activation='softmax')(x)
  else:
    outputs = layers.Dense(units=nlabels-1, kernel_initializer='he_normal', activation='sigmoid')(x)
  print('output', outputs.shape)

  model = Model(inputs, outputs, name="3dcnn")

  return model


def get_init(type='he_normal'):
               
  if type == 'he_normal':
      initial = tf.keras.initializers.HeNormal()
  elif type == 'he_uniform':
      initial = tf.keras.initializers.HeUniform()
  elif type == 'xavier_normal':
      initial = tf.keras.initializers.GlorotNormal()
  elif type == 'xavier_uniform':
      initial = tf.keras.initializers.GlorotUniform()
  else:
      raise ValueError('Unknown initialisation requested: %s' % type)
  
  return initial
               

def get_reg(type='None'):
   
  if type == 'L1':
      regularize = tf.keras.regularizers.L1()
  elif type == 'L2':
      regularize = tf.keras.regularizers.L2()
  elif type == 'L1L2':
      regularize = tf.keras.regularizers.L1L2()
  elif type == 'None':
      regularize = None
  else:
      raise ValueError('Unknown regularization requested: %s' % type)
  
  return regularize

               
def get_model(images, nlabels, config):
  
  return config.model_handle(images, nlabels, config)

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Concatenate, add
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import Model, Input
from tensorflow.keras import regularizers

import os

def model1(input_size = (256,256,3)):
  input = Input(input_size)
  x = Conv2D(32, kernel_size=(3, 3), activation='relu', padding="same", kernel_initializer='he_normal')(input)
  x = Conv2D(32, kernel_size=(3, 3), activation='relu', padding="same", kernel_initializer='he_normal')(x)
  x = MaxPooling2D(pool_size=(2, 2))(x)
  x = BatchNormalization()(x)
  
  x = Conv2D(64, kernel_size=(3, 3), activation='relu', padding="same", kernel_initializer='he_normal')(x)
  x = Conv2D(64, kernel_size=(3, 3), activation='relu', padding="same", kernel_initializer='he_normal')(x)
  x = MaxPooling2D(pool_size=(2, 2))(x)
  x = BatchNormalization()(x)
  
  x = Conv2D(128, kernel_size=(3, 3), activation='relu', padding="same", kernel_initializer='he_normal')(x)
  x = Conv2D(128, kernel_size=(3, 3), activation='relu', padding="same", kernel_initializer='he_normal')(x)
  x = MaxPooling2D(pool_size=(2, 2))(x)
  x = BatchNormalization()(x)
  
  x = Flatten()(x)
  x = Dense(128, activation='relu', kernel_initializer='he_normal')(x)
  x = Dropout(0.3)(x)
  x = Dense(64, activation='relu', kernel_initializer='he_normal')(x)
  x = Dropout(0.3)(x)
  # Add a final sigmoid layer with 1 node for classification output
  output = Dense(1, activation='sigmoid')(x)
  model = Model(inputs=input, outputs=output)
  return model

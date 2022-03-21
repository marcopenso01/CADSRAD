import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Concatenate, add
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import Model, Input
from tensorflow.keras import regularizers
from tensorflow.keras.layers import GlobalAveragePooling2D, Reshape, multiply, Permute
from tensorflow.keras import backend as K

import os

def model1(input_size = (256,256,3)):
  input = Input(input_size)
  x = Conv2D(48, kernel_size=(3, 3), padding="same", kernel_initializer='he_normal')(input)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = selective_kernel_block(x)
  x = MaxPooling2D(pool_size=(2, 2))(x)
  
  x = Conv2D(64, kernel_size=(3, 3), padding="same", kernel_initializer='he_normal')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = selective_kernel_block(x)
  x = MaxPooling2D(pool_size=(2, 2))(x)
  
  x = Conv2D(64, kernel_size=(3, 3), padding="same", kernel_initializer='he_normal')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Conv2D(64, kernel_size=(3, 3), padding="same", kernel_initializer='he_normal')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = MaxPooling2D(pool_size=(2, 2))(x)
  
  x = Flatten()(x)
  x = Dense(96, kernel_initializer='he_normal')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Dropout(0.3)(x)
  x = Dense(32, kernel_initializer='he_normal')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Dropout(0.3)(x)
  # Add a final sigmoid layer with 1 node for classification output
  output = Dense(1, activation='sigmoid')(x)
  model = Model(inputs=input, outputs=output)
  return model



def model2(input_size = (256,256,3)):
  input = Input(input_size)
  x = Conv2D(64, kernel_size=(3, 3), padding="same", kernel_initializer='he_normal')(input)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = spatial_attention(x)
  x = Conv2D(64, kernel_size=(3, 3), padding="same", kernel_initializer='he_normal')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = MaxPooling2D(pool_size=(2, 2))(x)
  
  x = Conv2D(128, kernel_size=(3, 3), padding="same", kernel_initializer='he_normal')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = spatial_attention(x)
  x = Conv2D(128, kernel_size=(3, 3), padding="same", kernel_initializer='he_normal')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = MaxPooling2D(pool_size=(2, 2))(x)
  
  x = Conv2D(128, kernel_size=(3, 3), padding="same", kernel_initializer='he_normal')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = spatial_attention(x)
  x = Conv2D(128, kernel_size=(3, 3), padding="same", kernel_initializer='he_normal')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = MaxPooling2D(pool_size=(2, 2))(x)
  
  x = GlobalAveragePooling2D()(x)
  # Add a final sigmoid layer with 1 node for classification output
  output = Dense(1, activation='sigmoid')(x)
  model = Model(inputs=input, outputs=output)
  return model

def model3(input_size = (256,256,3)):
  input = Input(input_size)
  x = Conv2D(48, kernel_size=(3, 3), padding="same", kernel_initializer='he_normal')(input)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Conv2D(48, kernel_size=(3, 3), padding="same", kernel_initializer='he_normal')(input)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = MaxPooling2D(pool_size=(2, 2))(x)
  
  x = Conv2D(64, kernel_size=(3, 3), padding="same", kernel_initializer='he_normal')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Conv2D(64, kernel_size=(3, 3), padding="same", kernel_initializer='he_normal')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = MaxPooling2D(pool_size=(2, 2))(x)
  
  x = Conv2D(128, kernel_size=(3, 3), padding="same", kernel_initializer='he_normal')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = channel_spatial_squeeze_excite(x, ratio=16)
  x = Conv2D(128, kernel_size=(3, 3), padding="same", kernel_initializer='he_normal')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = MaxPooling2D(pool_size=(2, 2))(x)
  
  x = GlobalAveragePooling2D()(x)
  # Add a final sigmoid layer with 1 node for classification output
  output = Dense(3, activation='softmax')(x)
  model = Model(inputs=input, outputs=output)
  return model


def se_block(input_tensor, ratio=16):
    """ Create a channel-wise squeeze-excite block
    Args:
        input_tensor: input Keras tensor
        ratio: number of output filters
    Returns: a Keras tensor
    References
    -   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
    """
    init = input_tensor
    channel = input_tensor.get_shape().as_list()[-1]

    se = GlobalAveragePooling2D()(init)
    se = Dense(channel // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(channel, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    x = multiply([init, se])
    return x


def spatial_attention(input_tensor):
    """ Create a spatial squeeze-excite block
    Args:
        input_tensor: input Keras tensor
    Returns: a Keras tensor
    References
    -   [Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks](https://arxiv.org/abs/1803.02579)
    """
    
    se = Conv2D(1, (1, 1), activation='sigmoid', use_bias=False,
                kernel_initializer='he_normal')(input_tensor)

    x = multiply([input_tensor, se])
    return x


def channel_spatial_squeeze_excite(input_tensor, ratio=16):
    """ Create a spatial squeeze-excite block
    Args:
        input_tensor: input Keras tensor
        ratio: number of output filters
    Returns: a Keras tensor
    References
    -   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
    -   [Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks](https://arxiv.org/abs/1803.02579)
    """

    cse = se_block(input_tensor, ratio)
    sse = spatial_attention(input_tensor)

    x = add([cse, sse])
    return x

def selective_kernel_block(input_tensor):
    channel = input_tensor.get_shape().as_list()[-1]
    d = int(channel / 2)
    init = input_tensor
    xs = []
    M=3
    for i in range(M):
        net = Conv2D(channel, (3,3), padding="SAME", dilation_rate=((i*i)+1, (i*i)+1),
                     kernel_initializer='he_normal', use_bias=False)(init)
        net = BatchNormalization()(net)
        net = Activation('relu')(net)

        xs.append(net)
    for i in range(M):
        if i == 0:
            U = xs[0]
        else:
            U = add([U, xs[i]])
    gap = tf.reduce_mean(U, [1, 2], keepdims=True)
    fc = Dense(d, kernel_initializer='he_normal', use_bias=False)(gap)
    fc = BatchNormalization()(fc)
    fc = Activation('sigmoid')(fc)
    att_vec = []

    for i in range(M):
        fcs = Dense(channel,kernel_initializer='he_normal', use_bias=False)(fc)
        fcs = Activation('softmax')(fcs)
        fea_v = multiply([fcs, xs[i]])

        att_vec.append(fea_v)
    for i in range(M):
        if i == 0:
            y = att_vec[0]
        else:
            y = add([y, att_vec[i]])
    return y

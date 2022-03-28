import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Concatenate, add
from tensorflow.keras.layers import Conv2D, MaxPooling2D, DepthwiseConv2D
from tensorflow.keras import Model, Input
from tensorflow.keras import regularizers
from tensorflow.keras.layers import GlobalAveragePooling2D, Reshape, multiply, Permute
from tensorflow.keras import backend as K
import numpy as np

import os


def model1(input_size=(256, 256, 3)):
    input = Input(input_size)
    x = Conv2D(48, kernel_size=(3, 3), padding="same", kernel_initializer='he_normal')(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(68, kernel_size=(3, 3), padding="same", kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(98, kernel_size=(3, 3), padding="same", kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = GlobalAveragePooling2D()(x)
    # Add a final sigmoid layer with 1 node for classification output
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=input, outputs=output)
    return model


def model2(input_size=(256, 256, 3)):
    input = Input(input_size)
    x = Conv2D(40, kernel_size=(3, 3), padding="same", kernel_initializer='he_normal')(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = residual_block(x, 60)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = residual_block(x, 80)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = GlobalAveragePooling2D()(x)
    # Add a final sigmoid layer with 1 node for classification output
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=input, outputs=output)
    return model


def model3(input_size=(256, 256, 3)):
    input = Input(input_size)
    x = Conv2D(48, kernel_size=(3, 3), padding="same", kernel_initializer='he_normal')(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(48, kernel_size=(3, 3), padding="same", kernel_initializer='he_normal')(x)
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

    x = Conv2D(92, kernel_size=(3, 3), padding="same", kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = spatial_attention(x)
    x = Conv2D(92, kernel_size=(3, 3), padding="same", kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = GlobalAveragePooling2D()(x)
    # Add a final sigmoid layer with 1 node for classification output
    output = Dense(3, activation='softmax')(x)
    model = Model(inputs=input, outputs=output)
    return model


def model4(input_size=(256, 256, 3)):
    """ConvMixer-256/8: https://openreview.net/pdf?id=TVHS5Y4dNvM.
    The hyperparameter values are taken from the paper.
    """
    input = Input(input_size)
    # Extract patch embeddings.
    x = conv_stem(input, filters=48, patch_size=2)

    # ConvMixer blocks.
    for _ in range(8):
        x = conv_mixer_block(x, filters=48, kernel_size=5)

    # Classification block.
    x = GlobalAveragePooling2D()(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=input, outputs=output)
    return model



def custom_gelu(x):
    return 0.5 * x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))

def activation_block(x):
    x = custom_gelu(x)
    return BatchNormalization()(x)

def conv_stem(x, filters: int, patch_size: int):
    x = Conv2D(filters, kernel_size=patch_size, strides=patch_size)(x)
    return activation_block(x)

def conv_mixer_block(x, filters: int, kernel_size: int):
    # Depthwise convolution.
    x0 = x
    x = DepthwiseConv2D(kernel_size=kernel_size, padding="same")(x)
    x = add([activation_block(x), x0])  # Residual.

    # Pointwise convolution.
    x = Conv2D(filters, kernel_size=1)(x)
    x = activation_block(x)
    return x

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
    M = 3
    for i in range(M):
        net = Conv2D(channel, (3, 3), padding="SAME", dilation_rate=((i * i) + 1, (i * i) + 1),
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
        fcs = Dense(channel, kernel_initializer='he_normal', use_bias=False)(fc)
        fcs = Activation('softmax')(fcs)
        fea_v = multiply([fcs, xs[i]])

        att_vec.append(fea_v)
    for i in range(M):
        if i == 0:
            y = att_vec[0]
        else:
            y = add([y, att_vec[i]])
    return y


def naive_inception_module(layer_in, f1):
    conv1 = Conv2D(f1, (3, 3), padding='same', activation='relu')(layer_in)

    conv2 = Conv2D(f1, (3, 3), padding='same', activation='relu')(layer_in)
    conv2 = Conv2D(f1, (3, 3), padding='same', activation='relu')(conv2)

    # concatenate filters, assumes filters/channels last
    layer_out = Concatenate(axis=1)([conv1, conv2])
    return layer_out

def residual_block(layer_in, f):
    y = layer_in
    x = Conv2D(f, kernel_size=(3, 3), padding="same", kernel_initializer='he_normal')(layer_in)
    x = BatchNormalization()(x)
    #x = Activation('relu')(x)
    #x = Conv2D(f, kernel_size=(3, 3), padding="same", kernel_initializer='he_normal')(x)
    #x = BatchNormalization()(x)
    y = Conv2D(f, kernel_size=(1, 1), padding="same", kernel_initializer='he_normal')(y)
    x = add([x, y])
    x = Activation('relu')(x)
    return x

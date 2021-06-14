"""
@author: Marco Penso
"""
import os
import numpy as np
import logging
import cv2
import shutil
from scipy import ndimage
import pandas as pd # for some simple data analysis (right now, just to load in the labels data and quickly reference it)
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras import layers 
from tensorflow.keras import Model 

import configuration as config
import read_data
import model_zoo as model_Zoo

# Set SGE_GPU environment
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

logging.basicConfig(
    level=logging.INFO # allow DEBUG level messages to pass through the logger
    )

log_dir = os.path.join(config.log_root, config.experiment_name)


def get_latest_model_checkpoint_path(folder, name):
    '''
    Returns the checkpoint with the highest iteration number with a given name
    :param folder: Folder where the checkpoints are saved
    :param name: Name under which you saved the model
    :return: The path to the checkpoint with the latest iteration
    '''

    iteration_nums = []
    for file in glob.glob(os.path.join(folder, '%s*.meta' % name)):

        file = file.split('/')[-1]
        file_base, postfix_and_number, rest = file.split('.')[0:3]
        it_num = int(postfix_and_number.split('-')[-1])

        iteration_nums.append(it_num)

    latest_iteration = np.max(iteration_nums)

    return os.path.join(folder, name + '-' + str(latest_iteration))


def run_training(continue_run):
        
        logging.info('EXPERIMENT NAME: %s' % config.experiment_name)
        
        init_step = 0
        
        if continue_run:
        logging.info('!!!!!!!!!!!!!!!!!!!!!!!!!!!! Continuing previous run !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        try:
            init_checkpoint_path = get_latest_model_checkpoint_path(log_dir, 'model.ckpt')
            logging.info('Checkpoint path: %s' % init_checkpoint_path)
            init_step = int(init_checkpoint_path.split('/')[-1].split('-')[-1]) + 1  # plus 1 b/c otherwise starts with eval
            logging.info('Latest step was: %d' % init_step)
        except:
            logging.warning('!!! Didnt find init checkpoint. Maybe first run failed. Disabling continue mode...')
            continue_run = False
            init_step = 0
        
        # load data
        data = read_data.load_and_maybe_process_data(
                input_folder=config.data_root,
                preprocessing_folder=config.preprocessing_folder,
                mode=config.data_mode,
                size=config.image_size,
                target_resolution=config.target_resolution,
                force_overwrite=False
            )
        
        # the following are HDF5 datasets, not numpy arrays
        imgs_train = data['data_train'][()]
        patient_train = data['patient_train'][()]
        label_train = data['classe_train'][()]

        if 'data_test' in data:
            imgs_val = data['data_test'][()]
            patient_val = data['patient_test'][()]
            label_val = data['classe_test'][()]
        
        logging.info('Data summary:')
        logging.info(' - Training Images:')
        logging.info(imgs_train.shape)
        logging.info(imgs_train.dtype)
        logging.info(' - Validation Images:')
        logging.info(imgs_val.shape)
        logging.info(imgs_val.dtype)
        
        # Build a model
        model, experiment_name = model_zoo.get_model(imgs_train, config)
        model.summary()
        
        if model.name in 'VGG16, InceptionV3, ResNet50, InceptionResNetV2, EfficientNetB0, EfficientNetB7, ResNet50V2' and config.data_mode == '3D':
            expand_dims = False   # (N,x,y,3)
        else:
            expand_dims = True
        
        for epoch in range(config.max_epochs):
            
            logging.info('EPOCH %d' % epoch)
            
            for batch in iterate_minibatches(imgs_train, 
                                             label_train,
                                             batch_size=config.batch_size,
                                             mode=config.data_mode,
                                             augment_batch=config.augment_batch,
                                             expand_dims):
                
                x, y = batch
               

def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 - 0.5
    o_y = float(y) / 2 - 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix
    
    
def apply_affine_transform(img, rows, cols, theta=0, tx=0, ty=0, 
                           fill_mode='nearest', order=1):
    '''
    Applies an affine transformation specified by the parameters given.
    :param img: A numpy array of shape [x, y, nchannels]
    :param rows: img rows
    :param cols: img cols
    :param theta: Rotation angle in degrees
    :param tx: Width shift
    :param ty: Heigh shift
    :param fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
    :param int, order of interpolation
    :return The transformed version of the input
    '''
    
    transform_matrix = None
    if theta != 0:
        theta = np.deg2rad(theta)
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        transform_matrix = rotation_matrix
    
    if tx != 0 or ty != 0:
        shift_matrix = np.array([[1, 0, tx],
                                 [0, 1, ty],
                                 [0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = shift_matrix
        else:
            transform_matrix = np.dot(transform_matrix, shift_matrix)
    
    if transform_matrix is not None:
        transform_matrix = transform_matrix_offset_center(
            transform_matrix, rows, cols)        
        final_affine_matrix = transform_matrix[:2, :2]
        final_offset = transform_matrix[:2, 2]
        
        #img = np.squeeze(img[...])  #(x,y)
        #img = np.rollaxis(img, 2, 0)
        channel_images = [ndimage.interpolation.affine_transform(
            img[:,:,channel],
            final_affine_matrix,
            final_offset,
            order=order,
            mode=fill_mode) for channel in range(img.shape[-1])]
        img = np.stack(channel_images, axis=2)
 
    return img
        
                
def augmentation_function(images, mode):
    '''
    Function for augmentation of minibatches. It will transform a set of images by 
    a number of optional transformations. Each image in the minibatch will be 
    seperately transformed with random parameters. 
    :param images: A numpy array of shape [minibatch, X, Y, (Z), nchannels]
    :param mode: Data mode (2D, 3D)
    :return: A mini batch of the same size but with transformed images. 
    '''
    
    new_images = []
    num_samples = images.shape[0]
    rows, cols = images.shape[1:3]
    
    for ii in range(num_samples):
        
        #img = np.squeeze(images[ii,...])
        img = images[ii,...]
        
        # RANDOM ROTATION
        if config.do_rotation_range is not False:
            theta = np.random.uniform(config.do_rotation_range[0], config.do_rotation_range[1])
        else:
            theta = 0
        
        #RANDOM WIDTH SHIFT
        width_rg = config.do_width_shift_range
        if width_rg is not False:
            #coin_flip = np.random.uniform(low=0.0, high=1.0)
            #if coin_flip > 0.5:
            if width_rg >= 1:
                ty = np.random.choice(int(width_rg))
                ty *= np.random.choice([-1, 1])
            elif width_rg >= 0 and width_rg < 1:
                ty = np.random.uniform(-width_rg,
                                       width_rg)
                ty = int(ty * cols)
            else:
                raise ValueError("do_width_shift_range parameter should be >0")
        else:
            ty = 0
            
        #RANDOM HEIGHT SHIFT
        height_rg = config.do_height_shift_range
        if height_rg is not False:
            #coin_flip = np.random.uniform(low=0.0, high=1.0)
            #if coin_flip > 0.5:
            if height_rg >= 1:
                tx = np.random.choice(int(height_rg))
                tx *= np.random.choice([-1, 1])
            elif height_rg >= 0 and height_rg < 1:
                tx = np.random.uniform(-height_rg,
                                       height_rg)
                tx = int(tx * rows)
            else:
                raise ValueError("do_height_shift_range parameter should be >0")
        else:
            tx = 0
            
        #RANDOM HORIZONTAL FLIP
        flip_horizontal = (np.random.random() < 0.5) * config.do_fliplr
              
        #RANDOM VERTICAL FLIP
        flip_vertical = (np.random.random() < 0.5) * config.do_flipud
        
        transform_parameters = {'theta': theta,
                                'tx': tx,
                                'ty': ty,
                                'flip_horizontal': flip_horizontal,
                                'flip_vertical': flip_vertical}
          
        img = apply_affine_transform(img, rows=rows, cols=cols,
                                     transform_parameters.get('theta', 0),
                                     transform_parameters.get('tx', 0),
                                     transform_parameters.get('ty', 0),
                                     fill_mode='nearest',
                                     order=1)
        
        if transform_parameters.get('flip_horizontal'):
            img = flip_axis(img, 1)

        if transform_parameters.get('flip_vertical'):
            img = flip_axis(img, 0)
    
        new_images.append(img)
    
    sampled_image_batch = np.asarray(new_images)
    
    return sampled_image_batch
    
    
def iterate_minibatches(images, labels, batch_size, mode, augment_batch=False, expand_dims=True):
    '''
    Function to create mini batches from the dataset of a certain batch size 
    :param images: tensor
    :param labels: tensor
    :param batch_size: batch size
    :param mode: data mode (2D, 3D)
    :param augment_batch: should batch be augmented?
    :param expand_dims: adding a dimension to a tensor?
    :return: mini batches
    '''

    random_indices = np.arange(images.shape[0])
    np.random.shuffle(random_indices)

    n_images = images.shape[0]

    for b_i in range(0,n_images,batch_size):

        if b_i + batch_size > n_images:
            continue
            
        batch_indices = np.sort(random_indices[b_i:b_i+batch_size])

        X = images[batch_indices, ...]
        y = labels[batch_indices]
        
        if expand_dims:        
            X = X[...,np.newaxis]   #array of shape [minibatch, X, Y, (Z), nchannels=1]

        if augment_batch:
            X = augmentation_function(X, mode)    

        yield X, y

        
def main():

    continue_run = True
    if not tf.io.gfile.exists(log_dir):
        tf.io.gfile.makedirs(log_dir)
        continue_run = False

    # Copy experiment config file
    shutil.copy(config.__file__, log_dir)

    run_training(continue_run)


if __name__ == '__main__':

    main()

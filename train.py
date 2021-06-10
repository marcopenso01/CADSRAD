"""
@author: Marco Penso
"""
import os
import numpy as np
import logging
import cv2
import shutil
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
        
        '''
        
        imgs_train = imgs_train[...,np.newaxis]
        if 'data_test' in data:
            imgs_val = imgs_val[...,np.newaxis]
            
        '''
        
        # Build a model
        model, experiment_name = model_zoo.get_model(imgs_train, config)
        model.summary()
        
        if model.name == 'VGG16': ....
        
        for epoch in range(config.max_epochs):
            
            logging.info('EPOCH %d' % epoch)
            
            for batch in iterate_minibatches(imgs_train, 
                                             label_train,
                                             batch_size=config.batch_size,
                                             augment_batch=config.augment_batch,
                                             experiment_name):
                
                x, y = batch
                

def augmentation_function(images):
    '''
    Function for augmentation of minibatches. It will transform a set of images by 
    a number of optional transformations. Each image in the minibatch will be 
    seperately transformed with random parameters. 
    :param images: A numpy array of shape [minibatch, X, Y, (Z), nchannels]
    :return: A mini batch of the same size but with transformed images. 
    '''
    
    if images.ndim = 3
    
                
def iterate_minibatches(images, labels, batch_size, augment_batch=False, experiment_name):
    '''
    Function to create mini batches from the dataset of a certain batch size 
    :param images: tensor
    :param labels: tensor
    :param batch_size: batch size
    :param augment_batch: should batch be augmented?
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

        #X = tf.expand_dims(X, -1)

        if augment_batch:
            X = augmentation_function(X)

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

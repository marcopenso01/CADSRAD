"""
@author: Marco Penso
"""
import os
import numpy as np
import logging
import cv2
import shutil
import math
import h5py
from scipy import ndimage
import pandas as pd # for some simple data analysis (right now, just to load in the labels data and quickly reference it)
import tensorflow as tf 
from keras.utils.np_utils import to_categorical
from tensorflow.keras import backend as K
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


def run_training(continue_run):
        
        logging.info('EXPERIMENT NAME: %s' % config.experiment_name)
        
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
        
        train_on_all_data = True
        
        # the following are HDF5 datasets, not numpy arrays
        imgs_train = data['data_train'][()]
        patient_train = data['patient_train'][()]
        label_train = data['classe_train'][()]

        if 'data_test' in data:
            
            train_on_all_data = False
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
        
        # Class Mapping 
        logging.info('CAD - 0:Normal | 1:1-24% | 2:25-49% | 3:50-69% | 4:70-99% | 5:100%')
        logging.info(dict(zip(unique, counts)))
        
        unique, counts = np.unique(label_train, return_counts=True)
        nlabels = len(unique)
        
        if (config.time_decay and (config.step_decay or config.exp_decay or config.adaptive_decay)) or (config.step_decay and (config.exp_decay or config.adaptive_decay)) or (config.exp_decay and config.adaptive_decay):
            raise AssertionError('Select a single learning rate decay')
        
        # Build a model
        model, experiment_name = model_zoo.get_model(imgs_train, nlabels, config)
        model.summary()
        
        #restore previous session
        if continue_run:
        logging.info('!!!!!!!!!!!!!!!!!!!!!!!!!!!! Continuing previous run !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        try:
            model.load_weights(os.path.join(log_dir, 'model_best_weights.h5'))
            logging.info('loading weights...')
            logging.info('Latest epoch was: %d' % init_step)
        except:
            logging.warning('!!! Didnt find init checkpoint. Maybe first run failed. Disabling continue mode...')
            continue_run = False
            init_step = 0
        
        if model.name in 'VGG16, InceptionV3, ResNet50, InceptionResNetV2, EfficientNetB0, EfficientNetB7, ResNet50V2' and config.data_mode == '3D':
            expand_dims = False   # (N,x,y,3)
        else:
            expand_dims = True
                    
        #METRICS
        if nlabels > 2:
            loss = tf.keras.losses.categorical_crossentropy
            metrics=[tf.keras.metrics.CategoricalAccuracy(), 
                     tf.keras.metrics.AUC(),
                     tf.keras.metrics.Recall(),
                     tf.keras.metrics.Precision()]
        else:
            loss = tf.keras.losses.binary_crossentropy   
            metrics=[tf.keras.metrics.BinaryAccuracy(), 
                     tf.keras.metrics.AUC(),
                     tf.keras.metrics.Recall(),
                     tf.keras.metrics.Precision()]
        
        curr_lr = config.learning_rate
        optimizer = tf.keras.optimizers.Adam(learning_rate=curr_lr, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
        
        logging.info('compiling model...')
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        
        history  = {}   #It records training metrics for each epoch
        val_history = {}    #It records validation metrics for each epoch
        lr_hist = []
        no_improvement_counter = 0
        last_train = np.inf
        best_val = np.inf
        step = init_step
        
        logging.info('Using TensorFlow backend')
        
        for epoch in range(config.max_epochs):
            
            logging.info('EPOCH %d/%d' % (epoch, onfig.max_epochs))
            
            temp_hist = {}   #It records training metrics for each batch
            
            for batch in iterate_minibatches(imgs_train, 
                                             label_train,
                                             nlabels,
                                             batch_size=config.batch_size,
                                             mode=config.data_mode,
                                             augment_batch=config.augment_batch,
                                             expand_dims):
                
                x, y = batch
                
                #TEMPORARY HACK (to avoid incomplete batches)
                if y.shape[0] < config.batch_size:
                    step += 1
                    continue
                
                hist = model.train_on_batch(x,y)
                
                #print("Train output: " + str(hist))
                
                if temp_hist == {}:
                    for m_i in range(len(model.metrics_names)):
                        temp_hist[model.metrics_names[m_i]] = []
                for key, i in zip(temp_hist, range(len(temp_hist))):
                    temp_hist[key].append(hist[i])
                
                if (step + 1) % config.step_train_eval_frequency == 0:
                    
                    train_loss = hist[0]
                    if train_loss <= last_train:  # best_train:
                        no_improvement_counter = 0
                        logging.info('Decrease in training error!)
                    else:
                        no_improvement_counter = no_improvement_counter+1
                        logging.info('No improvement in training error for %d steps' % no_improvement_counter)

                    last_train = train_loss
                     
                step += 1  #fine batch
            
                                     
            for key in temp_hist:
                temp_hist[key] = sum(temp_hist[key])/len(temp_hist[key])
            
            for m_k in range(len(model.metrics_names)):
                logging.info(str(model.metrics_names[m_k]+': %f') % temp_hist[model.metrics_names[m_k]])
            logging.info('Epoch: %d/%d' % (epoch, config.max_epochs))
            
            if history == {}:
                for m_i in range(len(model.metrics_names)):
                    history[model.metrics_names[m_i]] = []
            for key in history:
                history[key].append(temp_hist[key])
            
            #save learning rate history
            lr_hist.append(curr_lr)
                                     
            #decay learning rate
            if config.time_decay:
                #decay_rate = config.learning_rate / config.max_epochs
                decay_rate = 1E-4
                curr_lr *= (1. / (1. + decay_rate * epoch))
                K.set_value(model.optimizer.learning_rate, curr_lr)
            elif config.step_decay:
                drop = 0.5
                epochs_drop = 40.0
                curr_lr = config.learning_rate * math.pow(drop,
                          math.floor((1+epoch)/epochs_drop))
                K.set_value(model.optimizer.learning_rate, curr_lr)
            elif config.exp_decay:
                k = 0.01
                curr_lr = config.learning_rate * math.exp(-k*epoch)
                K.set_value(model.optimizer.learning_rate, curr_lr)
            elif config.adaptive_decay:
                curr_lr = config.learning_rate * temp_hist['loss']
                K.set_value(model.optimizer.learning_rate, curr_lr)
                                     
            logging.info('Current learning rate: %f' % curr_lr)
                                     
            # Save a checkpoint and evaluate the model against the validation set
            if not train_on_all_data:
                                     
                logging.info('Validation Data Eval:')
                val_hist = do_eval(imgs_val, label_val, nlabels,
                                   batch_size=config.batch_size,
                                   mode=config.data_mode,
                                   augment_batch=False,
                                   expand_dims)
                
                if val_history == {}:
                    for m_i in range(len(model.metrics_names)):
                        val_history[model.metrics_names[m_i]] = []
                for key, ii in zip(val_history, range(len(val_history))):
                    val_history[key].append(val_hist[ii])
                
                #save best model
                if val_hist[0] < best_val:
                    logging.info('val_loss improved from %f to %f, saving model to weights-improvement' % (best_val, val_hist[0]))
                    best_val = val_hist[0]
                    remove_file(log_dir)
                    #Weights-only saving
                    model.save_weights(os.path.join(log_dir, ('model_best_weights_' + str(round_up(best_val, 6)) + '_epoch_' + str(epoch) + '.h5')))
                    #Whole-model saving (configuration + weights)
                    #model.save(os.path.join(log_dir, 'best_model'))
                else:
                    logging.info('val_acc did not improve from %f' % best_val)
            
        #plot history (loss and metrics)
        for m_k in range(len(model.metrics_names)):
            plt.plot(history[model.metrics_names[m_k]], label=model.metrics_names[m_k])
            plt.plot(val_history[model.metrics_names[m_k]], label=str('val_' + model.metrics_names[m_k]))
            plt.title(str('model '+ model.metrics_names[m_k]))
            plt.legend()
            plt.xlabel('epoch')
            plt.ylabel(model.metrics_names[m_k])
            plt.show()
        #plot learning rate
        plt.plot(lr_hist)
        plt.title('model learning rate')
        plt.xlabel('epoch')
        plt.ylabel('learning rate')
        plt.show() 


def remove_file(my_dir):
    for fname in os.listdir(my_dir):
        if fname.startswith('model_best_weights_'):
            os.remove(os.path.join(my_dir, fname))


def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier
                                     
                                     
def do_eval(images, labels, nlabels, batch_size, mode, augment_batch=False, expand_dims=True):                           
    '''
    Function for running the evaluations on the validation sets.  
    :param images: A numpy array containing the images
    :param labels: A numpy array containing the corresponding labels 
    :param nlabels: number of labels
    :param batch_size: batch size
    :param mode: data mode (2D, 3D)
    :param augment_batch: should batch be augmented?
    :param expand_dims: adding a dimension to a tensor? 
    :return: Scalar val loss and metrics
    '''
    num_batches = 0
    history = []
    
    for batch in iterate_minibatches(images, 
                                     labels,
                                     nlabels,
                                     batch_size,
                                     mode,
                                     augment_batch,
                                     expand_dims):
        x, y = batch
                                     
        if y.shape[0] < batch_size:
            continue
        
        val_hist = model.test_on_batch(x,y)
        
        if history == []:
            history.append(val_hist)
        else:
            history[0] = [x + y for x, y in zip(history[0], val_hist)]
        num_batches += 1
    
    for i in range(len(history[0])):
        history[0][i] /= num_batches
                                     
    return history[0]
                               
                                     
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
                           fill_mode='constant', order=1):
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
            mode=fill_mode,
            cval=0.0) for channel in range(img.shape[-1])]
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
        
        flag=0
        if img.ndim > 3:    #(x,y,z,1)
            img = np.squeeze(img[...])
            flag=1
        
        img = apply_affine_transform(img, rows=rows, cols=cols,
                                     transform_parameters.get('theta', 0),
                                     transform_parameters.get('tx', 0),
                                     transform_parameters.get('ty', 0),
                                     fill_mode='constant',
                                     order=1)
        
        if transform_parameters.get('flip_horizontal'):
            img = flip_axis(img, 1)

        if transform_parameters.get('flip_vertical'):
            img = flip_axis(img, 0)
        
        if flag:
          img = img[...,np.newaxis]
        
        new_images.append(img)
    
    sampled_image_batch = np.asarray(new_images)
    
    return sampled_image_batch
    
    
def iterate_minibatches(images, labels, nlabels, batch_size, mode, augment_batch=False, expand_dims=True):
    '''
    Function to create mini batches from the dataset of a certain batch size 
    :param images: tensor
    :param labels: tensor
    :param nlabels: number of labels
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
        
        if nlabels > 2:
            y = to_categorical(y, nlabels)  #one-hot
        
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

import model_structure
import tensorflow as tf
import os
import socket
import logging

# Paths settings
data_root = '/content/drive/My Drive/Pazienti/train'      
test_data_root = '/content/drive/My Drive/Pazieni/test'
preprocessing_folder = '/content/drive/My Drive/preproc_data'     
project_root = '/content/drive/My Drive'                       
log_root = os.path.join(project_root, 'acdc_logdir')
weights_root = os.path.join(log_root, experiment_name)

# Data settings
data_mode = '2D'   #2D or 3D
image_size = (212, 212)
target_resolution = (1, 1)

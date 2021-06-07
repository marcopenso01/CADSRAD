"""
@author: Marco Penso
"""
import os

experiment_name = 'prova1'

# Model settings
model_handle = model_zoo.VGG16model
#model_handle = model_zoo.ResNet50V2model

# Paths settings
data_root = '/content/drive/My Drive/Pazienti/train'      
test_data_root = '/content/drive/My Drive/Pazieni/test'
preprocessing_folder = '/content/drive/My Drive/preproc_data'     
project_root = '/content/drive/My Drive'                       
log_root = os.path.join(project_root, 'acdc_logdir')
weights_root = os.path.join(log_root, experiment_name)

# Data settings
data_mode = '2D'   #2D or 3D
image_size = (212, 212)    #(212,212) or (116,116,28)  --> (nx,ny,Nz_max) with Nz_max = 0 padding is not applied. This might result in volumes with different Nz
target_resolution = (1, 1)   #(1.36719, 1.36719) or (2.5,2.5,5)
nlabels = 2

# Training settings
split_val_train = 0.2 # between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split.
batch_size = 8
learning_rate = 0.001


# Pre-process settings
standardize = False
normalize = True

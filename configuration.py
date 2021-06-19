"""
@author: Marco Penso
"""
import os
import model_zoo
import tensorflow as tf

experiment_name = 'prova1'

# Model settings
model_handle = model_zoo.VGG16_model  
#model_handle = model_zoo.InceptionV3_model
#model_handle = model_zoo.ResNet50_model
#model_handle = model_zoo.InceptionResNetV2_model
#model_handle = model_zoo.EfficientNetB0_model
#model_handle = model_zoo.EfficientNetB7_model
#model_handle = model_zoo.ResNet50V2_model

# fully-connected layer at the top of the network (for VGG16, Inception, ResNet and EfficientNet)
dense_layer = (512, 512)  #Number of filters for each dense layer: example (512,512)--> two dense layer of 512 filters
drop_rate = (0.5, 0.5)  # Dropout: example (0.5, 0)--> 1° dense layer drop_rate 0.5, 2° dense layer no dropout
kernel_init = 'he_normal'    # he_normal, he_uniform, xavier_uniform, xavier_normal
kernel_reg = 'None'     #L1, L2, L1L2, None

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

# Training settings
split_val_train = 0.2 # between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split.
batch_size = 8
learning_rate = 0.001   # initial learning rate value
max_epochs = 1000

#Decay Learning rate
time_decay = False     # LearningRate = LearningRate * 1/(1 + decay * epoch)
step_decay = False     # LearningRate = InitialLearningRate * DropRate^floor(epoch / epochDrop)
exp_decay = False      # LearningRate = InitialLearningRate * exp^(-decay * epoch)
adaptive_decay = False # LearningRate = InitialLearningRate * cost_function

# Augmentation settings
augment_batch = True    # should batch be augmented?
do_rotation_range = (-15, 15)  #random rotation in range (min,max), otherwise False
do_fliplr = False              #True: flip array in the left/right direction, otherwise False
do_flipud = False              #True: flip array in the up/down direction, otherwise False
do_width_shift_range = False   #fraction of total width, if < 1, or pixels if >= 1, otherwise False
do_height_shift_range = False  #fraction of total width, if < 1, or pixels if >= 1, otherwise False

# Pre-process settings
standardize = False
normalize = True

step_train_eval_frequency = 10

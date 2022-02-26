import numpy as np 
import os
import h5py
import skimage.io as io
import skimage.transform as trans
import matplotlib.pyplot as plt
from scipy import ndimage
import tensorflow as tf
from tensorflow.keras.utils import plot_model
import logging
from sklearn.utils import shuffle
import model_structure
import losses
logging.basicConfig(
    level=logging.INFO # allow DEBUG level messages to pass through the logger
    )

def standardize_image(image):
    '''
    make image zero mean and unit standard deviation
    '''

    img_o = np.float32(image.copy())
    m = np.mean(img_o)
    s = np.std(img_o)
    return np.divide((img_o - m), s)


def normalize_image(image):
    '''
    make image normalize between 0 and 1
    '''
    img_o = np.float32(image.copy())
    img_o = (img_o-img_o.min())/(img_o.max()-img_o.min())
    return img_o
  
 
def makefolder(folder):
    '''
    Helper function to make a new folder if doesn't exist
    :param folder: path to new folder
    :return: True if folder created, False if folder already exists
    '''
    if not os.path.exists(folder):
        os.makedirs(folder)
        return True
    return False
  
  
def train_test_split(img_data, cad_data, paz_data, ramo_data):
  '''
  function to create a cross validation from the dataset
  '''
  for k in range(5):
    #ciclo su k-fold
    coor_test = []
    for cad in np.unique(cad_data):
        paz = paz_data[np.where(cad_data==cad)]
        
        random_paz = shuffle(np.unique(paz), random_state=14)
        test_paz = random_paz[round(len(np.unique(paz))/5)*k:(round(len(np.unique(paz))/5)*k)+round(len(np.unique(paz))/5)]
        
        for i in test_paz:
            coor = np.where((cad_data==cad) & (paz_data==i))
            for j in range(len(coor[0][:])):
                coor_test.append(coor[0][j])
    
    train_img = []
    train_cad = []
    train_ramo = []
    test_img = []
    test_cad = []
    test_ramo = []
    
    for i in range(len(paz_data)):
        if i in coor_test:
            test_img.append(img_data[i])
            test_cad.append(cad_data[i])
            test_ramo.append(ramo_data[i])
        else:
            train_img.append(img_data[i])
            train_cad.append(cad_data[i])
            train_ramo.append(ramo_data[i])
            
    yield train_img, train_cad, train_ramo, test_img, test_cad, test_ramo
  
  
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
PATH
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
input_folder = '......'
output_folder = '......'

if not os.path.exists(input_folder):
  raise TypeError('no input path found %s' % input_folder)
if not os.path.exists(output_folder):
  makefolder(output_folder)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
LOAD TRAIN DATA
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
logging.info('\nLoading data...')
data = h5py.File(os.path.join(input_folder,'data.hdf5'), 'r')
img_data = data['img'][()]
cad_data = data['cad'][()]
paz_data = data['paz'][()]
ramo_data = data['ramo'][()]



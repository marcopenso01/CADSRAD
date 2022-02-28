import numpy as np 
import os
import h5py
import skimage.io as io
import skimage.transform as trans
from skimage import exposure
import matplotlib.pyplot as plt
from scipy import ndimage
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import 
import logging
import random
from sklearn.utils import shuffle
import model_zoo as model_zoo
from packaging import version
from tensorflow.python.client import device_lib
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
    
    tr_img = []
    tr_cad = []
    tr_ramo = []
    tr_paz = []
    test_img = []
    test_cad = []
    test_ramo = []
    
    for i in range(len(paz_data)):
        if i in coor_test:
            test_img.append(img_data[i])
            test_cad.append(cad_data[i])
            test_ramo.append(ramo_data[i])
        else:
            tr_img.append(img_data[i])
            tr_cad.append(cad_data[i])
            tr_ramo.append(ramo_data[i])
            tr_paz.append(paz_data[i])
    
    tr_img = np.asarray(tr_img)
    tr_cad = np.asarray(tr_cad)
    tr_ramo = np.asarray(tr_ramo)
    tr_paz = np.asarray(tr_paz)
    test_img = np.asarray(test_img)
    test_cad = np.asarray(test_cad)
    test_ramo = np.asarray(test_ramo)
    
    coor_test = []
    #validatio-train
    for cad in np.unique(tr_cad):
        paz = tr_paz[np.where(tr_cad==cad)]
        # /10 = 10% for validation
        var = round(len(np.unique(paz_data[np.where(cad_data==cad)]))/10)
        random_paz = shuffle(np.unique(paz), random_state=14)
        test_paz = random_paz[0:var]
        
        for i in test_paz:
            coor = np.where((tr_cad==cad) & (tr_paz==i))
            for j in range(len(coor[0][:])):
                coor_test.append(coor[0][j])
    
    val_img = []
    val_cad = []
    val_ramo = []
    train_img = []
    train_cad = []
    train_ramo = []
    
    for i in range(len(tr_paz)):
        if i in coor_test:
            val_img.append(tr_img[i])
            val_cad.append(tr_cad[i])
            val_ramo.append(tr_ramo[i])
        else:
            train_img.append(tr_img[i])
            train_cad.append(tr_cad[i])
            train_ramo.append(tr_ramo[i])
    
    train_img = np.asarray(train_img)
    train_cad = np.asarray(train_cad)
    train_ramo = np.asarray(train_ramo)
    val_img = np.asarray(val_img)
    val_cad = np.asarray(val_cad)
    val_ramo = np.asarray(val_ramo)
    tr_img = []
    tr_cad = []
    tr_paz = []
    tr_ramo = []
    
    yield train_img, train_cad, train_ramo, test_img, test_cad, test_ramo, val_img, val_cad, val_ramo
  

def zoom(img, zoom_factor):

    height, width = img.shape[:2] # It's also the final desired shape
    new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)

    ### Crop only the part that will remain in the result (more efficient)
    # Centered bbox of the final desired size in resized (larger/smaller) image coordinates
    y1, x1 = max(0, new_height - height) // 2, max(0, new_width - width) // 2
    y2, x2 = y1 + height, x1 + width
    bbox = np.array([y1,x1,y2,x2])
    # Map back to original image coordinates
    bbox = (bbox / zoom_factor).astype(np.int)
    y1, x1, y2, x2 = bbox
    cropped_img = img[y1:y2, x1:x2]
    # Handle padding when downscaling
    resize_height, resize_width = min(new_height, height), min(new_width, width)
    pad_height1, pad_width1 = (height - resize_height) // 2, (width - resize_width) //2
    pad_height2, pad_width2 = (height - resize_height) - pad_height1, (width - resize_width) - pad_width1
    pad_spec = [(pad_height1, pad_height2), (pad_width1, pad_width2)] + [(0,0)] * (img.ndim - 2)

    result = cv2.resize(cropped_img, (resize_width, resize_height))
    result = np.pad(result, pad_spec, mode='constant')
    assert result.shape[0] == height and result.shape[1] == width
    return result


def augmentation_function(images):
    '''
    Function for augmentation of minibatches.
    :param images: A numpy array of shape [minibatch, X, Y, nchannels]
    :return: A mini batch of the same size but with transformed images and masks. 
    '''
    new_images = []
    num_images = images.shape[0]
    channels = images.shape[-1]
    for ii in range(num_images):
        img = images[ii,...]
        # FLIP  up/down
        for ch in range(channels):
            if np.random.randint(2):
                img[...,ch] = np.flipud(img[...,ch])
        # RANDOM GAMMA CORRECTION
        gamma = random.randrange(8,13,1)
        img = exposure.adjust_gamma(img, gamma/10)
        # ZOOM
        if np.random.randint(2):
            img = zoom(img, round(random.uniform(0.97,1.03), 2))
        
        new_images.append(img)
    sampled_image_batch = np.asarray(new_images)
    return sampled_image_batch


def iterate_minibatches(images, labels, batch_size, augment_batch=False, expand_dims=True):
    '''
    Function to create mini batches from the dataset of a certain batch size 
    :param images: input data shape (N, W, H)
    :param labels: label data
    :param batch_size: batch size (Int)
    :param augment_batch: should batch be augmented?, Boolean (default: False)
    :param expand_dims: adding a dimension, Boolean (default: True)
    :return: mini batches
    '''
    n_images = images.shape[0]
    random_indices = np.arange(n_images)
    np.random.shuffle(random_indices)
    for b_i in range(0,n_images,batch_size):
        if b_i + batch_size > n_images:
            continue
        batch_indices = np.sort(random_indices[b_i:b_i+batch_size])
        X = images[batch_indices, ...]

        if augment_batch:
            X = augmentation_function(X)
        if expand_dims:        
            X = X[...,np.newaxis]   #array of shape [minibatch, X, Y, nchannels]
        
        yield X, y
        

def get_f1(y_true, y_pred):
    TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    Positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = TP / (Positives+K.epsilon())
    Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = TP / (Pred_Positives+K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val


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
LOAD DATA
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
logging.info('\nLoading data...')
file = h5py.File(os.path.join(input_folder,'data.hdf5'), 'r')
img_data = file['img'][()]
cad_data = file['cad'][()]
paz_data = file['paz'][()]
ramo_data = file['ramo'][()]
file.close()
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
HYPERPARAMETERS
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
batch_size = 2
epochs = 100
curr_lr = 1e-3
input_size = img_data[0].shape
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
TRAINING 
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#itero su k-fold
k_fold = 0
for data in train_test_split(img_data, cad_data, paz_data, ramo_data):
    train_img, train_cad, train_ramo, test_img, test_cad, test_ramo, val_img, val_cad, val_ramo = data
    print('-' * 70)
    print('---- Starting fold %d ----'% k_fold)
    print('-' * 70)
    
    out_fold = os.path.join(output_folder, str('fold'+k_fold))
    if not os.path.exists(out_fold):
        makefolder(out_fold)
        out_file = os.path.join(out_fold, 'summary_report.txt')
        with open(out_file, "w") as text_file:
            text_file.write('\n\n--------------------------------------------------------------------------\n')
            text_file.write('Model summary\n')
            text_file.write('----------------------------------------------------------------------------\n\n')
    
    print('training data', train_img.shape, train_img[0].dtype)
    print('validation data', val_img.shape, val_img[0].dtype)
    print('testing data', test_img.shape, test_img[0].dtype)
    
    print('\nCreating and compiling model...')
    model = model_zoo.model1(input_size = input_size)
    
    with open(out_file, "a") as text_file:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: text_file.write(x + '\n'))
    
    opt = Adam(learning_rate=curr_lr)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy',
                                                                      get_f1])
    print('model prepared...')
    print('Start training...')
    
    step = 0
    no_improvement_counter = 0
    train_history  = {}   #It records training metrics for each epoch
    val_history = {}    #It records validation metrics for each epoch
    for epoch in range(epochs):
        temp_hist = {}
        print('Epoch %d/%d' % (epoch+1, epochs))
        for batch in iterate_minibatches(train_img,
                                         train_cad,
                                         batch_size=batch_size,
                                         augment_batch=True,
                                         expand_dims=True):
            x, y = batch
            #TEMPORARY HACK (to avoid incomplete batches)
            if y.shape[0] < batch_size:
                step += 1
                continue
            
            hist = model.train_on_batch(x,y)
            
            if temp_hist == {}:
                for m_i in range(len(model.metrics_names)):
                    temp_hist[model.metrics_names[m_i]] = []
                
            for key, i in zip(temp_hist, range(len(temp_hist))):
                    temp_hist[key].append(hist[i])
            
            if (step + 1) % 20 == 0:
                logging.info(str('step: %d '+name_metric[0]+': %.3f '+name_metric[1]+': %.3f '+name_metric[2]+': %.3f '
                +name_metric[3]+': %.3f '+name_metric[4]+': %.3f '+name_metric[5]+': %.3f') % 
                             (step+1, hist[0], hist[1], hist[2], hist[3], hist[4], hist[5]))
        
            step += 1  #fine batch

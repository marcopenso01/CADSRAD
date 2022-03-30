import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
# for GPU process:
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import numpy as np
import h5py
from skimage import exposure
from matplotlib import pyplot as plt
from scipy import ndimage
import cv2
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam, SGD
import logging
import random
from sklearn.utils import shuffle
from sklearn import metrics
import pandas as pd
from sklearn.utils import class_weight
from scipy import stats
# import tensorflow_addons as tfa

import model_zoo as model_zoo
from packaging import version
from tensorflow.python.client import device_lib

logging.basicConfig(
    level=logging.INFO  # allow DEBUG level messages to pass through the logger
)

assert 'GPU' in str(device_lib.list_local_devices())

print('is_gpu_available: %s' % tf.test.is_gpu_available())  # True/False
# Or only check for gpu's with cuda support
print('gpu with cuda support: %s' % tf.test.is_gpu_available(cuda_only=True))
# tf.config.list_physical_devices('GPU') #The above function is deprecated in tensorflow > 2.1

print("TensorFlow version: ", tf.__version__)
assert version.parse(tf.__version__).release[0] >= 2, \
    "this notebook requires Tensorflow 2.0 or above"


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
    img_o = (img_o - img_o.min()) / (img_o.max() - img_o.min())
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
        # ciclo su k-fold
        coor_test = []
        for cad in np.unique(cad_data):
            paz = paz_data[np.where(cad_data == cad)]

            random_paz = shuffle(np.unique(paz), random_state=14)
            test_paz = random_paz[round(len(np.unique(paz)) / 5) * k:(round(len(np.unique(paz)) / 5) * k) + round(
                len(np.unique(paz)) / 5)]

            for i in test_paz:
                coor = np.where((cad_data == cad) & (paz_data == i))
                for j in range(len(coor[0][:])):
                    coor_test.append(coor[0][j])

        tr_img = []
        tr_cad = []
        tr_ramo = []
        tr_paz = []
        test_img = []
        test_cad = []
        test_ramo = []
        test_pt = []

        for i in range(len(paz_data)):
            if i in coor_test:
                test_img.append(img_data[i])
                test_cad.append(cad_data[i])
                test_ramo.append(ramo_data[i])
                test_pt.append(paz_data[i])
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
        test_pt = np.asarray(test_pt)

        coor_test = []
        # validatio-train
        for cad in np.unique(tr_cad):
            paz = tr_paz[np.where(tr_cad == cad)]
            # /10 = 10% for validation
            var = round(len(np.unique(paz_data[np.where(cad_data == cad)])) / 10)
            random_paz = shuffle(np.unique(paz), random_state=14)
            test_paz = random_paz[0:var]

            for i in test_paz:
                coor = np.where((tr_cad == cad) & (tr_paz == i))
                for j in range(len(coor[0][:])):
                    coor_test.append(coor[0][j])

        val_img = []
        val_cad = []
        val_ramo = []
        val_pt = []
        train_img = []
        train_cad = []
        train_ramo = []
        train_pt = []

        for i in range(len(tr_paz)):
            if i in coor_test:
                val_img.append(tr_img[i])
                val_cad.append(tr_cad[i])
                val_ramo.append(tr_ramo[i])
                val_pt.append(tr_paz[i])
            else:
                train_img.append(tr_img[i])
                train_cad.append(tr_cad[i])
                train_ramo.append(tr_ramo[i])
                train_pt.append(tr_paz[i])

        train_img = np.asarray(train_img)
        train_cad = np.asarray(train_cad)
        train_ramo = np.asarray(train_ramo)
        train_pt = np.asarray(train_pt)
        val_img = np.asarray(val_img)
        val_cad = np.asarray(val_cad)
        val_ramo = np.asarray(val_ramo)
        val_pt = np.asarray(val_pt)
        tr_img = []
        tr_cad = []
        tr_paz = []
        tr_ramo = []

        yield train_img, train_cad, train_ramo, test_img, test_cad, test_ramo, val_img, val_cad, val_ramo, train_pt, val_pt, test_pt


def zoom(img, zoom_factor):
    height, width = img.shape[:2]  # It's also the final desired shape
    new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)

    ### Crop only the part that will remain in the result (more efficient)
    # Centered bbox of the final desired size in resized (larger/smaller) image coordinates
    y1, x1 = max(0, new_height - height) // 2, max(0, new_width - width) // 2
    y2, x2 = y1 + height, x1 + width
    bbox = np.array([y1, x1, y2, x2])
    # Map back to original image coordinates
    bbox = (bbox / zoom_factor).astype(np.int)
    y1, x1, y2, x2 = bbox
    cropped_img = img[y1:y2, x1:x2]
    # Handle padding when downscaling
    resize_height, resize_width = min(new_height, height), min(new_width, width)
    pad_height1, pad_width1 = (height - resize_height) // 2, (width - resize_width) // 2
    pad_height2, pad_width2 = (height - resize_height) - pad_height1, (width - resize_width) - pad_width1
    pad_spec = [(pad_height1, pad_height2), (pad_width1, pad_width2)] + [(0, 0)] * (img.ndim - 2)

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
        img = images[ii, ...]
        # FLIP  up/down
        for ch in range(channels):
            if np.random.randint(2):
                img[..., ch] = np.flipud(img[..., ch])
        # RANDOM GAMMA CORRECTION
        gamma = random.randrange(8, 13, 1)
        img = exposure.adjust_gamma(img, gamma / 10)
        # ZOOM
        if np.random.randint(2):
            img = zoom(img, round(random.uniform(0.97, 1.03), 2))
        # CHANNELS SHUFFLE
        random_indices = np.arange(img.shape[-1])
        np.random.shuffle(random_indices)
        np.random.shuffle(random_indices)
        img_sh = img[..., [random_indices]]
        img_sh = np.squeeze(img_sh)

        new_images.append(img_sh)
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
    images = images / 255.0
    n_images = images.shape[0]
    random_indices = np.arange(n_images)
    np.random.shuffle(random_indices)
    for b_i in range(0, n_images, batch_size):
        if b_i + batch_size > n_images:
            continue
        batch_indices = np.sort(random_indices[b_i:b_i + batch_size])
        X = images[batch_indices, ...]
        y = labels[batch_indices, ...]

        if augment_batch:
            X = augmentation_function(X)
        if expand_dims:
            X = X[..., np.newaxis]  # array of shape [minibatch, X, Y, nchannels]

        yield X, y


def do_eval(images, labels, batch_size, augment_batch=False, expand_dims=True):
    '''
    Function for running the evaluations on the validation sets.  
    :param images: A numpy array containing the images
    :param labels: A numpy array containing the corresponding labels 
    :param batch_size: batch size
    :param expand_dims: adding a dimension to a tensor? 
    :return: Scalar val loss and metrics
    '''
    num_batches = 0
    history = []
    for batch in iterate_minibatches(images,
                                     labels,
                                     batch_size,
                                     augment_batch,
                                     expand_dims):
        x, y = batch
        if y.shape[0] < batch_size:
            continue

        val_hist = model.test_on_batch(x, y)
        if history == []:
            history.append(val_hist)
        else:
            history[0] = [x + y for x, y in zip(history[0], val_hist)]
        num_batches += 1

    for i in range(len(history[0])):
        history[0][i] /= num_batches

    return history[0]


def get_f1(y_true, y_pred):
    TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    Positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = TP / (Positives + K.epsilon())
    Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = TP / (Pred_Positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val


def print_txt(output_dir, stringa):
    out_file = os.path.join(output_dir, 'summary_report.txt')
    with open(out_file, "a") as text_file:
        text_file.writelines(stringa)


# apply threshold to positive probabilities to create labels
def to_labels(pos_probs, threshold):
    return (pos_probs >= threshold).astype('int')


def adjusted_classes(y_scores, t):
    """
    This function adjusts class predictions based on the prediction threshold (t).
    Will only work for binary classification problems.
    """
    return [1 if y >= t else 0 for y in y_scores]


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
PATH
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
input_folder = 'D:\CADSRAD-main\data'
output_folder = 'D:\CADSRAD-main\output\cad0_2\ex6'

if not os.path.exists(input_folder):
    raise TypeError('no input path found %s' % input_folder)
if not os.path.exists(output_folder):
    makefolder(output_folder)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
LOAD DATA
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
logging.info('\nLoading data...')
file = h5py.File(os.path.join(input_folder, 'data.hdf5'), 'r')
img_data = file['img'][()]
cad_data = file['cad'][()]
paz_data = file['paz'][()]
ramo_data = file['ramo'][()]
file.close()
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
ENHANCEMENT
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
'''
clahe = cv2.createCLAHE(clipLimit=1.5)
for pp in range(len(img_data)):
    for kk in range(4):
        img_data[pp,:,:,kk] = clahe.apply(img_data[pp,:,:,kk])
'''
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
NORMALIZATION
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
img_data = np.float32(img_data)
# img_data = img_data / 255.0
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
HYPERPARAMETERS
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
batch_size = 4
epochs = 600
curr_lr = 1e-3
input_size = img_data[0].shape

filters = 68
kernel_size1 = 2
stride_size = 2
kernel_size2 = 5
act = 'gelu'
batch_norm_pos = 'after'
blocks = 10
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
TRAINING 
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
PPV = []
NPV = []
ACC = []
BACC = []
RECALL = []
AUC = []
FPR = []
TPR = []

# itero su k-fold
k_fold = 0
for data in train_test_split(img_data, cad_data, paz_data, ramo_data):
    train_img, train_cad, train_ramo, test_img, test_cad, test_ramo, val_img, val_cad, val_ramo, train_paz, val_paz, test_paz = data
    print('-' * 70)
    print('---- Starting fold %d ----' % k_fold)
    print('-' * 70)

    '''
    # set classes: cad0-2 vs cad3-5
    train_cad[train_cad < 3] = 0
    train_cad[train_cad > 0] = 1
    val_cad[val_cad < 3] = 0
    val_cad[val_cad > 0] = 1
    '''

    # set classes: cad0 vs cad1-2
    train_img = train_img[np.where(train_cad < 3)[0][0]:np.where(train_cad < 3)[0][-1] + 1]
    train_ramo = train_ramo[np.where(train_cad < 3)[0][0]:np.where(train_cad < 3)[0][-1] + 1]
    train_paz = train_paz[np.where(train_cad < 3)[0][0]:np.where(train_cad < 3)[0][-1] + 1]
    train_cad = train_cad[np.where(train_cad < 3)[0][0]:np.where(train_cad < 3)[0][-1] + 1]
    val_img = val_img[np.where(val_cad < 3)[0][0]:np.where(val_cad < 3)[0][-1] + 1]
    val_ramo = val_ramo[np.where(val_cad < 3)[0][0]:np.where(val_cad < 3)[0][-1] + 1]
    val_paz = val_paz[np.where(val_cad < 3)[0][0]:np.where(val_cad < 3)[0][-1] + 1]
    val_cad = val_cad[np.where(val_cad < 3)[0][0]:np.where(val_cad < 3)[0][-1] + 1]
    test_img = test_img[np.where(test_cad < 3)[0][0]:np.where(test_cad < 3)[0][-1] + 1]
    test_ramo = test_ramo[np.where(test_cad < 3)[0][0]:np.where(test_cad < 3)[0][-1] + 1]
    test_paz = test_paz[np.where(test_cad < 3)[0][0]:np.where(test_cad < 3)[0][-1] + 1]
    test_cad = test_cad[np.where(test_cad < 3)[0][0]:np.where(test_cad < 3)[0][-1] + 1]
    train_cad[train_cad > 0] = 1
    val_cad[val_cad > 0] = 1
    '''
    # set classes: cad3-4 vs cad5
    print(val_cad)
    train_img = train_img[np.where(train_cad > 2)[0][0]:np.where(train_cad > 2)[0][-1] + 1]
    train_ramo = train_ramo[np.where(train_cad > 2)[0][0]:np.where(train_cad > 2)[0][-1] + 1]
    train_paz = train_paz[np.where(train_cad > 2)[0][0]:np.where(train_cad > 2)[0][-1] + 1]
    train_cad = train_cad[np.where(train_cad > 2)[0][0]:np.where(train_cad > 2)[0][-1] + 1]
    val_img = val_img[np.where(val_cad > 2)[0][0]:np.where(val_cad > 2)[0][-1] + 1]
    val_ramo = val_ramo[np.where(val_cad > 2)[0][0]:np.where(val_cad > 2)[0][-1] + 1]
    val_paz = val_paz[np.where(val_cad > 2)[0][0]:np.where(val_cad > 2)[0][-1] + 1]
    val_cad = val_cad[np.where(val_cad > 2)[0][0]:np.where(val_cad > 2)[0][-1] + 1]
    test_img = test_img[np.where(test_cad > 2)[0][0]:np.where(test_cad > 2)[0][-1] + 1]
    test_ramo = test_ramo[np.where(test_cad > 2)[0][0]:np.where(test_cad > 2)[0][-1] + 1]
    test_paz = test_paz[np.where(test_cad > 2)[0][0]:np.where(test_cad > 2)[0][-1] + 1]
    test_cad = test_cad[np.where(test_cad > 2)[0][0]:np.where(test_cad > 2)[0][-1] + 1]
    train_cad[train_cad <= 4] = 0
    val_cad[val_cad <= 4] = 0
    train_cad[train_cad == 5] = 1
    val_cad[val_cad == 5] = 1
    print(val_cad)
    '''

    # set weights classes
    # class_weights = class_weight.compute_class_weight(class_weight='balanced',classes=np.unique(train_cad),y=train_cad)
    # class_weights = dict(zip(np.unique(train_cad), class_weights))

    # class_weight = {0: 0.3,
    #                1: 0.7}

    out_fold = os.path.join(output_folder, 'fold' + str(k_fold))
    if not os.path.exists(out_fold):
        makefolder(out_fold)
        out_file = os.path.join(out_fold, 'summary_report.txt')
        with open(out_file, "w") as text_file:
            text_file.write('\n\n--------------------------------------------------------------------------\n')
            text_file.write('Model summary\n')
            text_file.write('----------------------------------------------------------------------------\n\n')

    print('Training data', train_img.shape, train_img[0].dtype)
    print_txt(out_fold, ['\nTraining data %d' % len(train_img)])
    print('Validation data', val_img.shape, val_img[0].dtype)
    print_txt(out_fold, ['\nValidation data %d' % len(val_img)])

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    MODEL
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    print('\nCreating and compiling model...')
    model = model_zoo.model4(input_size=input_size, filters=filters, kernel_size1=kernel_size1, stride_size=stride_size,
                             kernel_size2=kernel_size2, act=act, batch=batch_norm_pos, blocks=blocks)

    print_txt(out_fold, ['\n\nHyperParameters:'])
    print_txt(out_fold, ['\nfilters %s' % filters])
    print_txt(out_fold, ['\nkernel_size1 %s' % kernel_size1])
    print_txt(out_fold, ['\nstride_size %s' % stride_size])
    print_txt(out_fold, ['\nkernel_size2 %s' % kernel_size2])
    print_txt(out_fold, ['\nact %s' % act])
    print_txt(out_fold, ['\nbatch_norm_pos %s' % batch_norm_pos])
    print_txt(out_fold, ['\nblocks %s' % blocks])
    print_txt(out_fold, ['\nbatch_size %s\n' % batch_size])

    with open(out_file, "a") as text_file:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: text_file.write(x + '\n'))

    # opt = Adam(learning_rate=curr_lr)
    opt = SGD(learning_rate=curr_lr, momentum=0.9)
    # opt = tfa.optimizers.AdamW(learning_rate=0.001, weight_decay=0.0001)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    print('Model prepared...')
    print('Start training...')

    step = 0
    no_improvement_counter = 0
    best_val_loss = float('inf')
    train_history = {}  # It records training metrics for each epoch
    val_history = {}  # It records validation metrics for each epoch
    for epoch in range(epochs):
        temp_hist = {}
        print('Epoch %d/%d' % (epoch + 1, epochs))
        for batch in iterate_minibatches(train_img,
                                         train_cad,
                                         batch_size=batch_size,
                                         augment_batch=True,
                                         expand_dims=False):
            x, y = batch
            # TEMPORARY HACK (to avoid incomplete batches)
            if y.shape[0] < batch_size:
                step += 1
                continue

            # hist = model.train_on_batch(x,y, class_weight=class_weights)
            hist = model.train_on_batch(x, y)

            if temp_hist == {}:
                for m_i in range(len(model.metrics_names)):
                    temp_hist[model.metrics_names[m_i]] = []

            for key, i in zip(temp_hist, range(len(temp_hist))):
                temp_hist[key].append(hist[i])

            if (step + 1) % 20 == 0:
                print('step: %d, %s: %.3f, %s: %.3f' %
                      (step + 1, model.metrics_names[0], hist[0], model.metrics_names[1], hist[1]))

            step += 1  # fine batch

        for key in temp_hist:
            temp_hist[key] = sum(temp_hist[key]) / len(temp_hist[key])

        print('Training data Eval')
        print('%s: %.3f, %s: %.3f' %
              (model.metrics_names[0], temp_hist[model.metrics_names[0]],
               model.metrics_names[1], temp_hist[model.metrics_names[1]]))

        if train_history == {}:
            for m_i in range(len(model.metrics_names)):
                train_history[model.metrics_names[m_i]] = []
        for key in train_history:
            train_history[key].append(temp_hist[key])

        print('Validation data Eval')
        val_hist = do_eval(val_img, val_cad,
                           batch_size=batch_size,
                           augment_batch=False,
                           expand_dims=False)

        if val_history == {}:
            for m_i in range(len(model.metrics_names)):
                val_history[model.metrics_names[m_i]] = []
        for key, ii in zip(val_history, range(len(val_history))):
            val_history[key].append(val_hist[ii])

        # save best model
        if val_hist[0] < best_val_loss:
            no_improvement_counter = 0
            print('val_loss improved from %.3f to %.3f, saving model to weights-improvement' % (
                best_val_loss, val_hist[0]))
            best_val_loss = val_hist[0]
            model.save(os.path.join(out_fold, 'model_weights.h5'))
            # model.save_weights(os.path.join(out_fold, 'model_weights.h5'))
        else:
            no_improvement_counter += 1
            print('val_loss did not improve for %d epochs' % no_improvement_counter)

        # ReduceLROnPlateau
        if no_improvement_counter % 6 == 0 and no_improvement_counter != 0:
            old_lr = curr_lr
            curr_lr = curr_lr * 0.2
            if curr_lr < 1e-6:
                curr_lr = 1e-4
            K.set_value(model.optimizer.learning_rate, curr_lr)
            print('Learning rate changed from %.6f to %.6f' % (old_lr, curr_lr))
        # EarlyStopping
        if no_improvement_counter > 34:  # Early stop if val loss does not improve after n epochs
            print('Early stop at epoch %d' % (epoch + 1))
            break

    print('\nModel correctly trained and saved')
    # Plot
    plt.figure(figsize=(8, 8))
    plt.grid(False)
    plt.title("Learning curve LOSS", fontsize=20)
    plt.plot(train_history["loss"], label="Loss")
    plt.plot(val_history["loss"], label="Validation loss")
    p = np.argmin(val_history["loss"])
    plt.plot(p, val_history["loss"][p], marker="x", color="r", label="best model")
    plt.xlabel("Epochs", fontsize=16)
    plt.ylabel("Loss", fontsize=16)
    plt.legend();
    plt.savefig(os.path.join(out_fold, 'Loss'), dpi=300)
    plt.close()

    # free memory
    del train_img
    del val_img

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    TESTING AND EVALUATING THE MODEL
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    print('-' * 50)
    print('Testing...')
    print('-' * 50)
    print('Testing data', test_img.shape, test_img[0].dtype)
    print_txt(out_fold, ['\nTesting data %d' % len(test_img)])
    print_txt(out_fold, ['\nTest patients %s' % test_paz])
    print_txt(out_fold, ['\nTest CAD class %s' % test_cad])
    test_img = test_img / 255.0
    cad_class = np.copy(test_cad)
    # test_cad[test_cad < 3] = 0
    test_cad[test_cad > 0] = 1
    # test_cad[test_cad <= 4] = 0
    # test_cad[test_cad == 5] = 1
    ''
    print_txt(out_fold, ['\nTest binary CAD class %s' % test_cad])
    print('Loading saved weights...')
    model = tf.keras.models.load_model(os.path.join(out_fold, 'model_weights.h5'))
    print('Predicting...')
    prediction = model.predict(test_img)

    # calculate roc curves
    fpr, tpr, thresholds = metrics.roc_curve(test_cad, prediction, pos_label=1)
    aucc = metrics.roc_auc_score(test_cad, prediction)
    # plot the roc curve for the model
    plt.figure()
    plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
    plt.plot(fpr, tpr, marker='.', label="ROC curve (area = %0.2f)" % aucc)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.savefig(os.path.join(out_fold, 'AUC'), dpi=300)
    plt.close()

    # define thresholds
    thresholds = np.arange(0, 1, 0.001)
    # evaluate each threshold
    scores = [metrics.f1_score(test_cad, to_labels(prediction, t)) for t in thresholds]
    # get best threshold
    ix = np.argmax(scores)
    print_txt(out_fold, ['\nThreshold=%.3f, F-Score=%.5f' % (thresholds[ix], scores[ix])])
    pred_adj = adjusted_classes(prediction, thresholds[ix])
    # precision
    # precision = metrics.precision_score(test_cad, pred_adj)
    # print_txt(out_fold, ['\nPrecision: %.2f' % precision])
    # recall
    # recall = metrics.recall_score(test_cad, pred_adj)
    # print_txt(out_fold, ['\nRecall: %.2f' % recall])
    # f1
    # f1 = metrics.f1_score(test_cad, pred_adj)
    # print_txt(out_fold, ['\nf1: %.2f' % f1])
    # ROC AUC
    print('ROC AUC: %f' % aucc)
    print_txt(out_fold, ['\nROC AUC: %f' % aucc])

    # print(metrics.classification_report(test_labels, pred_adj))
    print_txt(out_fold, ['\n\n %s \n\n' % metrics.classification_report(test_cad, pred_adj)])

    CM = metrics.confusion_matrix(test_cad, pred_adj)
    disp = metrics.ConfusionMatrixDisplay(CM)
    disp.plot()
    # plt.show()
    plt.savefig(os.path.join(out_fold, 'Conf_matrix'), dpi=300)
    plt.close()

    TN = CM[0][0]
    print_txt(out_fold, ['\ntrue negative: %d' % TN])
    FN = CM[1][0]
    print_txt(out_fold, ['\nfalse negative: %d' % FN])
    TP = CM[1][1]
    print_txt(out_fold, ['\ntrue positive: %d' % TP])
    FP = CM[0][1]
    print_txt(out_fold, ['\nfalse positive: %d' % FP])
    prec = TP / (TP + FP)
    print_txt(out_fold, ['\nPrecision or Pos predictive value: %.2f' % prec])
    rec = TP / (TP + FN)
    print_txt(out_fold, ['\nRecall: %.2f' % rec])
    print_txt(out_fold, ['\nSpecificity: %.2f' % (TN / (TN + FP))])
    print_txt(out_fold, ['\nNeg predictive value: %.2f' % (TN / (FN + TN))])
    print_txt(out_fold, ['\nF1: %.2f' % (2 * (prec * rec) / (prec + rec))])
    print_txt(out_fold, ['\nAcc: %.2f' % ((TP + TN) / (TN + FN + TP + FP))])
    print_txt(out_fold, ['\nBalanced_Acc: %.2f\n' % metrics.balanced_accuracy_score(test_cad, pred_adj)])

    PPV.append(prec)
    NPV.append((TN / (FN + TN)))
    ACC.append(((TP + TN) / (TN + FN + TP + FP)))
    BACC.append(metrics.balanced_accuracy_score(test_cad, pred_adj))
    RECALL.append(rec)
    AUC.append(aucc)
    FPR.append(fpr)
    TPR.append(tpr)

    with open(out_file, "a") as text_file:
        text_file.write('\n----- Prediction ----- \n')
        text_file.write('real_class       probability        pred_class      paz       ramo\n')
        for ii in range(len(prediction)):
            text_file.write(
                '%d                %.3f                 %d                  %d                %d\n' % (
                    test_cad[ii], prediction[ii], pred_adj[ii], test_paz[ii], test_ramo[ii]))

    df1 = pd.DataFrame({'labl': test_cad, 'pred': prediction[:, 0], 'paz': test_paz, 'ramo': test_ramo})
    df1.to_excel(os.path.join(out_fold, 'Excel_df1.xlsx'))

    # for patient
    with open(out_file, "a") as text_file:
        text_file.write('\n\n----- for patient ----- \n')
    true = []
    pred = []
    for c in np.unique(cad_class):
        for p in np.unique(test_paz):
            if len(np.where((test_paz == p) & (cad_class == c))[0][:]) != 0:
                true.append(test_cad[np.where((test_paz == p) & (cad_class == c))[0][0]])
                flag = 0
                for ii in range(len(np.where((test_paz == p) & (cad_class == c))[0])):
                    if pred_adj[np.where((test_paz == p) & (cad_class == c))[0][ii]] == 1:
                        flag = 1
                if flag:
                    pred.append(1)
                else:
                    pred.append(0)

    true = np.asarray(true)
    pred = np.asarray(pred)
    CM = metrics.confusion_matrix(true, pred)
    TN = CM[0][0]
    print_txt(out_fold, ['\ntrue negative: %d' % TN])
    FN = CM[1][0]
    print_txt(out_fold, ['\nfalse negative: %d' % FN])
    TP = CM[1][1]
    print_txt(out_fold, ['\ntrue positive: %d' % TP])
    FP = CM[0][1]
    print_txt(out_fold, ['\nfalse positive: %d' % FP])
    prec = TP / (TP + FP)
    print_txt(out_fold, ['\nPrecision or Pos predictive value: %.2f' % prec])
    rec = TP / (TP + FN)
    print_txt(out_fold, ['\nRecall: %.2f' % rec])
    print_txt(out_fold, ['\nSpecificity: %.2f' % (TN / (TN + FP))])
    print_txt(out_fold, ['\nNeg predictive value: %.2f' % (TN / (FN + TN))])
    print_txt(out_fold, ['\nF1: %.2f' % (2 * (prec * rec) / (prec + rec))])
    print_txt(out_fold, ['\nAcc: %.2f' % ((TP + TN) / (TN + FN + TP + FP))])
    print_txt(out_fold, ['\nBalanced_Acc: %.2f\n' % metrics.balanced_accuracy_score(true, pred)])

    k_fold += 1

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
EVALUATE MEAN PERFORMANCE
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
mean_ppv = np.mean(PPV)
mean_npv = np.mean(NPV)
mean_acc = np.mean(ACC)
mean_bacc = np.mean(BACC)
mean_rec = np.mean(RECALL)
mean_auc = np.mean(AUC)
std_ppv = np.std(PPV, ddof=1)
std_npv = np.std(NPV, ddof=1)
std_acc = np.std(ACC, ddof=1)
std_bacc = np.std(BACC, ddof=1)
std_rec = np.std(RECALL, ddof=1)
std_auc = np.std(AUC, ddof=1)

# formula is true for sample <30. In this case sample = 5
infer_ppv = round(mean_ppv - (round(stats.t.ppf(1 - 0.025, len(PPV) - 1), 3) * std_ppv / np.sqrt(len(PPV))), 4)
upper_ppv = round(mean_ppv + (round(stats.t.ppf(1 - 0.025, len(PPV) - 1), 3) * std_ppv / np.sqrt(len(PPV))), 4)
infer_npv = round(mean_npv - (round(stats.t.ppf(1 - 0.025, len(NPV) - 1), 3) * std_npv / np.sqrt(len(NPV))), 4)
upper_npv = round(mean_npv + (round(stats.t.ppf(1 - 0.025, len(NPV) - 1), 3) * std_npv / np.sqrt(len(NPV))), 4)
infer_acc = round(mean_acc - (round(stats.t.ppf(1 - 0.025, len(ACC) - 1), 3) * std_acc / np.sqrt(len(ACC))), 4)
upper_acc = round(mean_acc + (round(stats.t.ppf(1 - 0.025, len(ACC) - 1), 3) * std_acc / np.sqrt(len(ACC))), 4)
infer_bacc = round(mean_bacc - (round(stats.t.ppf(1 - 0.025, len(BACC) - 1), 3) * std_bacc / np.sqrt(len(BACC))), 4)
upper_bacc = round(mean_bacc + (round(stats.t.ppf(1 - 0.025, len(BACC) - 1), 3) * std_bacc / np.sqrt(len(BACC))), 4)
infer_rec = round(mean_rec - (round(stats.t.ppf(1 - 0.025, len(RECALL) - 1), 3) * std_rec / np.sqrt(len(RECALL))), 4)
upper_rec = round(mean_rec + (round(stats.t.ppf(1 - 0.025, len(RECALL) - 1), 3) * std_rec / np.sqrt(len(RECALL))), 4)
infer_auc = round(mean_auc - (round(stats.t.ppf(1 - 0.025, len(AUC) - 1), 3) * std_auc / np.sqrt(len(AUC))), 4)
upper_auc = round(mean_auc + (round(stats.t.ppf(1 - 0.025, len(AUC) - 1), 3) * std_auc / np.sqrt(len(AUC))), 4)
if upper_auc > 1:
    upper_auc = 1.0

print("Mean PPV = %0.2f (CI %0.2f-%0.2f)" % (mean_ppv, infer_ppv, upper_ppv))
print("Mean NPV = %0.2f (CI %0.2f-%0.2f)" % (mean_npv, infer_npv, upper_npv))
print("Mean ACC = %0.2f (CI %0.2f-%0.2f)" % (mean_acc, infer_acc, upper_acc))
print("Mean BACC = %0.2f (CI %0.2f-%0.2f)" % (mean_bacc, infer_bacc, upper_bacc))
print("Mean RECALL = %0.2f (CI %0.2f-%0.2f)" % (mean_rec, infer_rec, upper_rec))
print("Mean AUC = %0.2f (CI %0.2f-%0.2f)" % (mean_auc, infer_auc, upper_auc))

out_file = os.path.join(output_folder, 'summary_report.txt')
print_txt(output_folder, ['\nMean PPV = %0.2f (CI %0.2f-%0.2f)' % (mean_ppv, infer_ppv, upper_ppv)])
print_txt(output_folder, ['\nMean NPV = %0.2f (CI %0.2f-%0.2f)' % (mean_npv, infer_npv, upper_npv)])
print_txt(output_folder, ['\nMean ACC = %0.2f (CI %0.2f-%0.2f)' % (mean_acc, infer_acc, upper_acc)])
print_txt(output_folder, ['\nMean BACC = %0.2f (CI %0.2f-%0.2f)' % (mean_bacc, infer_bacc, upper_bacc)])
print_txt(output_folder, ['\nMean RECALL = %0.2f (CI %0.2f-%0.2f)' % (mean_rec, infer_rec, upper_rec)])
print_txt(output_folder, ['\nMean AUC = %0.2f (CI %0.2f-%0.2f)' % (mean_auc, infer_auc, upper_auc)])

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color="r", alpha=0.8)
plt.plot(FPR[0], TPR[0], lw=2, alpha=0.5, label="ROC fold 0 (AUC = %0.2f)" % AUC[0])
plt.plot(FPR[1], TPR[1], lw=2, alpha=0.5, label="ROC fold 1 (AUC = %0.2f)" % AUC[1])
plt.plot(FPR[2], TPR[2], lw=2, alpha=0.5, label="ROC fold 2 (AUC = %0.2f)" % AUC[2])
plt.plot(FPR[3], TPR[3], lw=2, alpha=0.5, label="ROC fold 3 (AUC = %0.2f)" % AUC[3])
plt.plot(FPR[4], TPR[4], lw=2, alpha=0.5, label="ROC fold 4 (AUC = %0.2f)" % AUC[4])
plt.plot([], [], ' ', label="Mean AUC = %0.2f (CI %0.2f-%0.2f)" % (mean_auc, infer_auc, upper_auc))
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.savefig(os.path.join(output_folder, 'CV_AUC'), dpi=1200)
plt.close()

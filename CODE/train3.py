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
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam, SGD
import logging
import random
from sklearn.utils import shuffle
from sklearn import metrics
import pandas as pd
from sklearn.utils import class_weight
from scipy import stats

import model_zoo as model_zoo
from packaging import version
from tensorflow.python.client import device_lib
from itertools import cycle

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


def plot_confusion_matrix(cm,
                          target_names,
                          path,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot
    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix
    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']
    title:        the text to display at the top of the matrix
    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues
    normalize:    If False, plot the raw numbers
                  If True, plot the proportions
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    # plt.show()
    plt.savefig(path, dpi=400, bbox_inches='tight')
    plt.close()


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
PATH
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
input_folder = 'D:\CADSRAD-main\data'
output_folder = 'D:\CADSRAD-main\output\cad3_5\ex1'
model_path = 'D:\CADSRAD-main\output\cad0_5\ex3'

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
NORMALIZATION
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
img_data = np.float32(img_data)
img_data = img_data / 255.0
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
HYPERPARAMETERS
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
batch_size = 4
epochs = 400
curr_lr = 1e-4
input_size = img_data[0].shape
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
TRAINING 
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
ACC = []
macroPPV = []
weightedPPV = []
macroNPV = []
weightedNPV = []
macroF1 = []
weightedF1 = []
macroRec = []
weightedRec = []
AUC0 = []
AUC1 = []
AUC2 = []

# itero su k-fold
k_fold = 0
for data in train_test_split(img_data, cad_data, paz_data, ramo_data):
    train_img, train_cad, train_ramo, test_img, test_cad, test_ramo, val_img, val_cad, val_ramo, train_paz, val_paz, test_paz = data
    print('-' * 70)
    print('---- Starting fold %d ----' % k_fold)
    print('-' * 70)

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

    print('classes CADRADS:', np.unique(test_cad))

    # ONE-HOT encoding
    train_cad[train_cad == 3] = 0
    train_cad[train_cad == 4] = 1
    train_cad[train_cad == 5] = 2
    val_cad[val_cad == 3] = 0
    val_cad[val_cad == 4] = 1
    val_cad[val_cad == 5] = 2

    train_cad = to_categorical(train_cad)
    val_cad = to_categorical(val_cad)

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

    # -------------- MODEL ---------------
    print('\nCreating and compiling model...')
    model = model_zoo.model3(input_size=input_size)
    # model = tf.keras.models.load_model(os.path.join(model_path, 'fold'+str(k_fold), 'model_weights.h5'))
    # x = model.layers[-2].output
    # x = Dropout(0.3)(x)
    # predictions = Dense(3, activation="softmax")(x)
    # model = Model(inputs = model.input, outputs = predictions)

    with open(out_file, "a") as text_file:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: text_file.write(x + '\n'))

    # opt = Adam(learning_rate=curr_lr)
    opt = SGD(learning_rate=curr_lr, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["categorical_accuracy"])
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
                      (step + 1, model.metrics_names[0], hist[0],
                       model.metrics_names[1], hist[1]))

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
            curr_lr = curr_lr * 0.5
            if curr_lr < 1e-6:
                curr_lr = 1e-5
            K.set_value(model.optimizer.learning_rate, curr_lr)
            print('Learning rate changed from %.6f to %.6f' % (old_lr, curr_lr))
        # EarlyStopping
        if no_improvement_counter > 35:  # Early stop if val loss does not improve after n epochs
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
    test_cad[test_cad == 3] = 0
    test_cad[test_cad == 4] = 1
    test_cad[test_cad == 5] = 2
    cad_class = np.copy(test_cad)

    print('Loading saved weights...')
    model = tf.keras.models.load_model(os.path.join(out_fold, 'model_weights.h5'))
    print('Predicting...')
    prediction = model.predict(test_img)

    y = np.argmax(prediction, axis=-1)

    # multi class confusion matrix
    cnf_matrix = metrics.confusion_matrix(test_cad, y)
    plt.figure()
    plot_confusion_matrix(cnf_matrix, target_names=['3', '4', '5'],
                          path=os.path.join(out_fold, 'Conf_matrix'),
                          title='Confusion matrix', normalize=False)

    # metrics
    NPV0 = (np.sum(cnf_matrix[1][1:]) + np.sum(cnf_matrix[2][1:])) / (
            np.sum(cnf_matrix[1][1:]) + np.sum(cnf_matrix[2][1:]) + np.sum(cnf_matrix[0][1:]))
    NPV1 = (cnf_matrix[0][0] + cnf_matrix[0][-1] + cnf_matrix[-1][0] + cnf_matrix[-1][-1]) / (
            cnf_matrix[0][0] + cnf_matrix[0][-1] + cnf_matrix[-1][0] + cnf_matrix[-1][-1] + cnf_matrix[1][0] +
            cnf_matrix[1][-1])
    NPV2 = (np.sum(cnf_matrix[0][0:2]) + np.sum(cnf_matrix[1][0:2])) / (
            np.sum(cnf_matrix[0][0:2]) + np.sum(cnf_matrix[1][0:2]) + np.sum(cnf_matrix[2][0:2]))
    n_class0 = len(np.where(test_cad == np.unique(test_cad)[0])[0])
    n_class1 = len(np.where(test_cad == np.unique(test_cad)[1])[0])
    n_class2 = len(np.where(test_cad == np.unique(test_cad)[2])[0])

    print_txt(out_fold, ['\n\n\nAccuracy: %.2f' % metrics.accuracy_score(test_cad, y)])

    print_txt(out_fold, ['\nMicro Precision (PPV): %.2f' % metrics.precision_score(test_cad, y, average='micro')])
    print_txt(out_fold, ['\nMicro Recall: %.2f' % metrics.recall_score(test_cad, y, average='micro')])
    print_txt(out_fold, ['\nMicro F1-score: %.2f' % metrics.f1_score(test_cad, y, average='micro')])

    print_txt(out_fold, ['\nMacro Precision (PPV): %.2f' % metrics.precision_score(test_cad, y, average='macro')])
    print_txt(out_fold, ['\nMacro Recall: %.2f' % metrics.recall_score(test_cad, y, average='macro')])
    print_txt(out_fold, ['\nMacro F1-score: %.2f' % metrics.f1_score(test_cad, y, average='macro')])
    print_txt(out_fold, ['\nMacro NPV: %.2f' % ((NPV0 + NPV1 + NPV2) / 3)])

    print_txt(out_fold, ['\nWeighted Precision (PPV): %.2f' % metrics.precision_score(test_cad, y, average='weighted')])
    print_txt(out_fold, ['\nWeighted Recall: %.2f' % metrics.recall_score(test_cad, y, average='weighted')])
    print_txt(out_fold, ['\nWeighted F1-score: %.2f' % metrics.f1_score(test_cad, y, average='weighted')])
    print_txt(out_fold,
              ['\nWeighted NPV: %.2f' % (((NPV0 * n_class0) + (NPV1 * n_class1) + (NPV2 * n_class2)) / len(test_cad))])

    print_txt(out_fold, [
        '\n\n %s \n\n' % metrics.classification_report(test_cad, y, target_names=['Class 3', 'Class 4', 'Class 5'])])

    # roc curve for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(3):
        fpr[i], tpr[i], _ = metrics.roc_curve(to_categorical(test_cad)[:, i], prediction[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    colors = cycle(["limegreen", "darkorange", "cornflowerblue"])
    plt.figure()
    for i, color in zip(range(3), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=2,
            label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic to multiclass")
    plt.legend(loc="lower right")
    # plt.show()
    plt.savefig(os.path.join(out_fold, 'AUC'), dpi=400)
    plt.close()

    ACC.append(metrics.accuracy_score(test_cad, y))
    macroPPV.append(metrics.precision_score(test_cad, y, average='macro'))
    weightedPPV.append(metrics.precision_score(test_cad, y, average='weighted'))
    macroNPV.append(((NPV0 + NPV1 + NPV2) / 3))
    weightedNPV.append((((NPV0 * n_class0) + (NPV1 * n_class1) + (NPV2 * n_class2)) / len(test_cad)))
    macroF1.append(metrics.f1_score(test_cad, y, average='macro'))
    weightedF1.append(metrics.f1_score(test_cad, y, average='weighted'))
    macroRec.append(metrics.recall_score(test_cad, y, average='macro'))
    weightedRec.append(metrics.recall_score(test_cad, y, average='weighted'))
    AUC0.append(roc_auc[0])
    AUC1.append(roc_auc[1])
    AUC2.append(roc_auc[2])

    with open(out_file, "a") as text_file:
        text_file.write('\n----- Prediction ----- \n')
        text_file.write(
            'real_class        probability                  pred_class               paz              ramo\n')
        for ii in range(len(prediction)):
            text_file.write(
                '%d                %.3f  %.3f  %.3f                 %d                  %d                %d\n' % (
                    test_cad[ii], prediction[ii][0], prediction[ii][1], prediction[ii][2], y[ii], test_paz[ii],
                    test_ramo[ii]))

    pred0 = []
    pred1 = []
    pred2 = []
    for ii in range(len(prediction)):
        pred0.append(prediction[ii][0])
        pred1.append(prediction[ii][1])
        pred2.append(prediction[ii][2])
    df1 = pd.DataFrame(
        {'real_class': test_cad, 'pred_class': y, 'pred_prob3': pred0, 'pred_prob4': pred1, 'pred_prob5': pred2,
         'paz': test_paz, 'ramo': test_ramo})
    df1.to_excel(os.path.join(out_fold, 'Excel_df1.xlsx'))

    # for patient
    with open(out_file, "a") as text_file:
        text_file.write('\n\n----- for patient ----- \n')
    count = 0
    summ = 0
    for c in np.unique(cad_class):
        for p in np.unique(test_paz):
            if len(np.where((test_paz == p) & (cad_class == c))[0][:]) != 0:
                flag = 0
                for ii in range(len(np.where((test_paz == p) & (cad_class == c))[0])):
                    if y[np.where((test_paz == p) & (cad_class == c))[0][ii]] != c:
                        flag = 1
                if flag == 0:
                    count += 1
                summ += 1
    print_txt(out_fold, ['\nAccuracy: %.2f' % (count / summ)])

    k_fold += 1

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
EVALUATE MEAN PERFORMANCE
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

mean_acc = np.mean(ACC)
std_acc = np.std(ACC, ddof=1)

mean_macroPPV = np.mean(macroPPV)
std_macroPPV = np.std(macroPPV, ddof=1)

mean_weightedPPV = np.mean(weightedPPV)
std_weightedPPV = np.std(weightedPPV, ddof=1)

mean_macroNPV = np.mean(macroNPV)
std_macroNPV = np.std(macroNPV, ddof=1)

mean_weightedNPV = np.mean(weightedNPV)
std_weightedNPV = np.std(weightedNPV, ddof=1)

mean_macroF1 = np.mean(macroF1)
std_macroF1 = np.std(macroF1, ddof=1)

mean_weightedF1 = np.mean(weightedF1)
std_weightedF1 = np.std(weightedF1, ddof=1)

mean_macroRec = np.mean(macroRec)
std_macroRec = np.std(macroRec, ddof=1)

mean_weightedRec = np.mean(weightedRec)
std_weightedRec = np.std(weightedRec, ddof=1)

mean_AUC0 = np.mean(AUC0)
std_AUC0 = np.std(AUC0, ddof=1)

mean_AUC1 = np.mean(AUC1)
std_AUC1 = np.std(AUC1, ddof=1)

mean_AUC2 = np.mean(AUC2)
std_AUC2 = np.std(AUC2, ddof=1)

# formula is true for sample <30. In this case sample = 5
infer_acc = round(mean_acc - (round(stats.t.ppf(1 - 0.025, len(ACC) - 1), 3) * std_acc / np.sqrt(len(ACC))), 4)
upper_acc = round(mean_acc + (round(stats.t.ppf(1 - 0.025, len(ACC) - 1), 3) * std_acc / np.sqrt(len(ACC))), 4)

infer_macroPPV = round(
    mean_macroPPV - (round(stats.t.ppf(1 - 0.025, len(macroPPV) - 1), 3) * std_macroPPV / np.sqrt(len(macroPPV))), 4)
upper_macroPPV = round(
    mean_macroPPV + (round(stats.t.ppf(1 - 0.025, len(macroPPV) - 1), 3) * std_macroPPV / np.sqrt(len(macroPPV))), 4)

infer_weightedPPV = round(mean_weightedPPV - (
        round(stats.t.ppf(1 - 0.025, len(weightedPPV) - 1), 3) * std_weightedPPV / np.sqrt(len(weightedPPV))), 4)
upper_weightedPPV = round(mean_weightedPPV + (
        round(stats.t.ppf(1 - 0.025, len(weightedPPV) - 1), 3) * std_weightedPPV / np.sqrt(len(weightedPPV))), 4)

infer_macroNPV = round(
    mean_macroNPV - (round(stats.t.ppf(1 - 0.025, len(macroNPV) - 1), 3) * std_macroNPV / np.sqrt(len(macroNPV))), 4)
upper_macroNPV = round(
    mean_macroNPV + (round(stats.t.ppf(1 - 0.025, len(macroNPV) - 1), 3) * std_macroNPV / np.sqrt(len(macroNPV))), 4)

infer_weightedNPV = round(mean_weightedNPV - (
        round(stats.t.ppf(1 - 0.025, len(weightedNPV) - 1), 3) * std_weightedNPV / np.sqrt(len(weightedNPV))), 4)
upper_weightedNPV = round(mean_weightedNPV + (
        round(stats.t.ppf(1 - 0.025, len(weightedNPV) - 1), 3) * std_weightedNPV / np.sqrt(len(weightedNPV))), 4)

infer_macroF1 = round(
    mean_macroF1 - (round(stats.t.ppf(1 - 0.025, len(macroF1) - 1), 3) * std_macroF1 / np.sqrt(len(macroF1))), 4)
upper_macroF1 = round(
    mean_macroF1 + (round(stats.t.ppf(1 - 0.025, len(macroF1) - 1), 3) * std_macroF1 / np.sqrt(len(macroF1))), 4)

infer_weightedF1 = round(mean_weightedF1 - (
        round(stats.t.ppf(1 - 0.025, len(weightedF1) - 1), 3) * std_weightedF1 / np.sqrt(len(weightedF1))), 4)
upper_weightedF1 = round(mean_weightedF1 + (
        round(stats.t.ppf(1 - 0.025, len(weightedF1) - 1), 3) * std_weightedF1 / np.sqrt(len(weightedF1))), 4)

infer_macroRec = round(
    mean_macroRec - (round(stats.t.ppf(1 - 0.025, len(macroRec) - 1), 3) * std_macroRec / np.sqrt(len(macroRec))), 4)
upper_macroRec = round(
    mean_macroRec + (round(stats.t.ppf(1 - 0.025, len(macroRec) - 1), 3) * std_macroRec / np.sqrt(len(macroRec))), 4)

infer_weightedRec = round(mean_weightedRec - (
        round(stats.t.ppf(1 - 0.025, len(weightedRec) - 1), 3) * std_weightedRec / np.sqrt(len(weightedRec))), 4)
upper_weightedRec = round(mean_weightedRec + (
        round(stats.t.ppf(1 - 0.025, len(weightedRec) - 1), 3) * std_weightedRec / np.sqrt(len(weightedRec))), 4)

infer_AUC0 = round(mean_AUC0 - (round(stats.t.ppf(1 - 0.025, len(AUC0) - 1), 3) * std_AUC0 / np.sqrt(len(AUC0))), 4)
upper_AUC0 = round(mean_AUC0 + (round(stats.t.ppf(1 - 0.025, len(AUC0) - 1), 3) * std_AUC0 / np.sqrt(len(AUC0))), 4)

infer_AUC1 = round(mean_AUC1 - (round(stats.t.ppf(1 - 0.025, len(AUC1) - 1), 3) * std_AUC1 / np.sqrt(len(AUC1))), 4)
upper_AUC1 = round(mean_AUC1 + (round(stats.t.ppf(1 - 0.025, len(AUC1) - 1), 3) * std_AUC1 / np.sqrt(len(AUC1))), 4)

infer_AUC2 = round(mean_AUC2 - (round(stats.t.ppf(1 - 0.025, len(AUC2) - 1), 3) * std_AUC2 / np.sqrt(len(AUC2))), 4)
upper_AUC2 = round(mean_AUC2 + (round(stats.t.ppf(1 - 0.025, len(AUC2) - 1), 3) * std_AUC2 / np.sqrt(len(AUC2))), 4)

print("Mean ACC = %0.2f (CI %0.2f-%0.2f)" % (mean_acc, infer_acc, upper_acc))
print("Mean macroPPV = %0.2f (CI %0.2f-%0.2f)" % (mean_macroPPV, infer_macroPPV, upper_macroPPV))
print("Mean weightedPPV = %0.2f (CI %0.2f-%0.2f)" % (mean_weightedPPV, infer_weightedPPV, upper_weightedPPV))
print("Mean macroNPV = %0.2f (CI %0.2f-%0.2f)" % (mean_macroNPV, infer_macroNPV, upper_macroNPV))
print("Mean weightedNPV = %0.2f (CI %0.2f-%0.2f)" % (mean_weightedNPV, infer_weightedNPV, upper_weightedNPV))
print("Mean macroF1 = %0.2f (CI %0.2f-%0.2f)" % (mean_macroF1, infer_macroF1, upper_macroF1))
print("Mean weightedF1 = %0.2f (CI %0.2f-%0.2f)" % (mean_weightedF1, infer_weightedF1, upper_weightedF1))
print("Mean macroRec = %0.2f (CI %0.2f-%0.2f)" % (mean_macroRec, infer_macroRec, upper_macroRec))
print("Mean weightedRec = %0.2f (CI %0.2f-%0.2f)" % (mean_weightedRec, infer_weightedRec, upper_weightedRec))
print("Mean AUC0 = %0.2f (CI %0.2f-%0.2f)" % (mean_AUC0, infer_AUC0, upper_AUC0))
print("Mean AUC1 = %0.2f (CI %0.2f-%0.2f)" % (mean_AUC1, infer_AUC1, upper_AUC1))
print("Mean AUC2 = %0.2f (CI %0.2f-%0.2f)" % (mean_AUC2, infer_AUC2, upper_AUC2))

out_file = os.path.join(output_folder, 'summary_report.txt')
print_txt(output_folder, ['\nMean ACC = %0.2f (CI %0.2f-%0.2f)' % (mean_acc, infer_acc, upper_acc)])
print_txt(output_folder, ['\nMean macroPPV = %0.2f (CI %0.2f-%0.2f)' % (mean_macroPPV, infer_macroPPV, upper_macroPPV)])
print_txt(output_folder,
          ['\nMean weightedPPV = %0.2f (CI %0.2f-%0.2f)' % (mean_weightedPPV, infer_weightedPPV, upper_weightedPPV)])
print_txt(output_folder, ['\nMean macroNPV = %0.2f (CI %0.2f-%0.2f)' % (mean_macroNPV, infer_macroNPV, upper_macroNPV)])
print_txt(output_folder,
          ['\nMean weightedNPV = %0.2f (CI %0.2f-%0.2f)' % (mean_weightedNPV, infer_weightedNPV, upper_weightedNPV)])
print_txt(output_folder, ['\nMean macroF1 = %0.2f (CI %0.2f-%0.2f)' % (mean_macroF1, infer_macroF1, upper_macroF1)])
print_txt(output_folder,
          ['\nMean weightedF1 = %0.2f (CI %0.2f-%0.2f)' % (mean_weightedF1, infer_weightedF1, upper_weightedF1)])
print_txt(output_folder, ['\nMean macroRec = %0.2f (CI %0.2f-%0.2f)' % (mean_macroRec, infer_macroRec, upper_macroRec)])
print_txt(output_folder,
          ['\nMean weightedRec = %0.2f (CI %0.2f-%0.2f)' % (mean_weightedRec, infer_weightedRec, upper_weightedRec)])
print_txt(output_folder, ['\nMean AUC0 = %0.2f (CI %0.2f-%0.2f)' % (mean_AUC0, infer_AUC0, upper_AUC0)])
print_txt(output_folder, ['\nMean AUC1 = %0.2f (CI %0.2f-%0.2f)' % (mean_AUC1, infer_AUC1, upper_AUC1)])
print_txt(output_folder, ['\nMean AUC2 = %0.2f (CI %0.2f-%0.2f)' % (mean_AUC2, infer_AUC2, upper_AUC2)])

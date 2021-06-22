"""
@author: Marco Penso
"""
import os
import numpy as np
import logging
import cv2
import pandas as pd # for some simple data analysis (right now, just to load in the labels data and quickly reference it)
import tensorflow as tf
import sklearn.metrics as metrics
from sklearn.preprocessing import label_binarize
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

  
def get_latest_model_checkpoint(my_dir):
    for fname in os.listdir(my_dir):
      if fname.startswith('model_best_loss_'):
          return fname

        
def test_data(input_folder, output_folder, model_path, config, gt_exists=True):
  
  logging.warning('EVALUATING ON TEST SET')
  
  # load data
  data = read_data.load_and_maybe_process_data(
         input_folder=input_folder,
         preprocessing_folder=config.preprocessing_folder,
         mode=config.data_mode,
         size=config.image_size,
         target_resolution=config.target_resolution,
         train=False,
         force_overwrite=False
  )
  
  # the following are HDF5 datasets, not numpy arrays
  imgs_test = data['data_train'][()]
  patient_test = data['patient_train'][()]
  label_test = data['classe_train'][()]

  if 'data_test' in data:
    logging.warning('ATTENTION: DATA SPLIT INTO TRAIN AND TEST SET...')
    logging.warning('THIS SHOULD NOT HAPPEN!!!')

  unique, counts = np.unique(patient_test, return_counts=True)
  npat = len(unique)
  
  logging.info('Data summary:')
  logging.info(' - Test Images:')
  logging.info(imgs_test.shape)
  logging.info(imgs_test.dtype)
  logging.info('Number of patients to evaluate %d' % npat)
  
  #restore previous session
  try:
    best_model_file = get_latest_model_checkpoint(model_path)
    model = load_model(os.path.join(log_dir, best_model_file)) ## returns a compiled model
    logging.info('loading model...')
  except:
    raise AssertionError('Impossible restore a compiled model')
  
  nclasses = model.output.shape[1]
    
  ytrue = []
  ypred = []
  ypred_prob = []
  
  for ii in range(len(imgs_test[0])):
    
    img = imgs_test[ii,...]
    label = label_test[ii,...]
    
    if not model.name in 'VGG16, InceptionV3, ResNet50, InceptionResNetV2, EfficientNetB0, EfficientNetB7, ResNet50V2' and config.data_mode == '3D':
      img = img[...,np.newaxis]
    
    prediction = model.predict(np.expand_dims(img, axis=0))
    
    ytrue.append(label)
    ypred.append(np.argmax(prediction))
    ypred_prob.append(prediction[0].tolist())
  
  ypred_prob = np.array(ypred_prob, dtype=float)
  
  ytrue_bin = label_binarize(ytrue, classes= range(nclasses))
  
  logging.info('Accuracy classification score: %f' % metrics.accuracy_score(ytrue, ypred))
  logging.info('Balanced accuracy: %f' % metrics.balanced_accuracy_score(ytrue, ypred))
  logging.info('Precision: %f' % metrics.precision_score(ytrue, ypred, average='micro'))
  logging.info('Weighted precision: %f' % metrics.precision_score(ytrue, ypred, average='weighted'))
  logging.info('Recall: %f' % metrics.recall_score(ytrue, ypred, average='micro'))
  logging.info('Weighted recall: %f' % metrics.recall_score(ytrue, ypred, average='weighted'))
  logging.info('F1-score: %f' % metrics.f1_score(ytrue, ypred, average='micro'))
  logging.info('Weighted F1-score: %f' % metrics.f1_score(ytrue, ypred, average='weighted'))
  
  logging.info('confusion matrix:')
  print(metrics.confusion_matrix(ytrue, ypred))
  
  fpr = dict()
  tpr = dict()
  roc_auc = dict()
  for i in range(nclasses):
    fpr[i], tpr[i], _ = metrics.roc_curve(ytrue_bin[:, i], ypred_prob[:, i])
    roc_auc[i] = metrics.auc(fpr[i], tpr[i])
  
  colors = ['b', 'g', 'r', 'c', 'm', 'y']
  for i, color in zip(range(nclasses), colors):
      plt.plot(fpr[i], tpr[i], color=color, 
               label='ROC curve of class {0} (area = {1:0.2f})'
               ''.format(i, roc_auc[i]))
  plt.plot([0, 1], [0, 1], 'k--')
  plt.xlim([-0.05, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('ROC curve')
  plt.legend(loc="lower right")
  plt.show()
  
if __name__ == '__main__':
  
  model_path = os.path.join(config.log_root, config.experiment_name)
  input_path = config.test_data_root
  output_path = os.path.join(model_path, 'predictions')
  
  makefolder(output_path)
  
  test_data(input_path, output_path, model_path, config=config)
  

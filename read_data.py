import os
import glob
import numpy as np
import logging
import nibabel as nib
import gc
import h5py
from skimage import transform
from skimage import util
import cv2
from PIL import Image
import pydicom

import configuration as config

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
  
def crop_or_pad_slice_to_size(slice, nx, ny):

    x, y = slice.shape

    x_s = (x - nx) // 2
    y_s = (y - ny) // 2
    x_c = (nx - x) // 2
    y_c = (ny - y) // 2

    if x > nx and y > ny:
        slice_cropped = slice[x_s:x_s + nx, y_s:y_s + ny]
    else:
        slice_cropped = np.zeros((nx, ny))
        if x <= nx and y > ny:
            slice_cropped[x_c:x_c + x, :] = slice[:, y_s:y_s + ny]
        elif x > nx and y <= ny:
            slice_cropped[:, y_c:y_c + y] = slice[x_s:x_s + nx, :]
        else:
            slice_cropped[x_c:x_c + x, y_c:y_c + y] = slice[:, :]

    return slice_cropped
  
  
  def prepare_data((input_folder, output_file, mode, size, target_resolution):
    
    '''
    Main function that prepares a dataset from the raw challenge data to an hdf5 dataset
    '''
    assert (mode in ['2D', '3D']), 'Unknown mode: %s' % mode
    if mode == '2D' and not len(size) == 2:
        raise AssertionError('Inadequate number of size parameters')
    if mode == '3D' and not len(size) == 3:
        raise AssertionError('Inadequate number of size parameters')
    if mode == '2D' and not len(target_resolution) == 2:
        raise AssertionError('Inadequate number of target resolution parameters')
    if mode == '3D' and not len(target_resolution) == 3:
        raise AssertionError('Inadequate number of target resolution parameters')
    
    hdf5_file = h5py.File(output_file, "w")
    data_addrs = []
                   
    if mode == '2D':
        nx, ny = size
                   
    elif mode == '3D':
        nx, ny, nz_max = size  
    
    for CAD in sorted(os.listdir(input_folder)):
        CAD_path = os.path.join(input_folder, CAD)
        class_cad =  CAD.split(' rads ')[1]
                   
        for fold in os.listdir(CAD_path):
            fold_path = os.path.join(CAD_path, fold)
                   
                   for PAZ in sorted(os.listdir(fold_path)):
                       PAZ_path = os.path.join(fold_path, PAZ)
                       
                       for file in sorted(os.listdir(PAZ_path)):
                           dcmPath = os.path.join(PAZ_path, file)
                           data_row_img = pydicom.dcmread(dcmPath)
                           image = np.int16(data_row_img.pixel_array)
                           
    
    n_data = len(data_addrs)
                          
    
    # Create datasets
    for i in range(len(data_addrs)):
        
                   
  def load_and_maybe_process_data(input_folder,
                                  preprocessing_folder,
                                  mode,
                                  size,
                                  target_resolution,
                                  force_overwrite=False):
    
    '''
    This function is used to load and if necessary preprocesses the dataset
    
    :param input_folder: Folder where the raw data is located 
    :param preprocessing_folder: Folder where the proprocessed data should be written to
    :param mode: Can either be '2D' or '3D'
    :param size: Size of the output slices/volumes in pixels/voxels
    :param target_resolution: Resolution to which the data should resampled. Should have same shape as size
    :param force_overwrite: Set this to True if you want to overwrite already preprocessed data [default: False]
     
    :return: Returns an h5py.File handle to the dataset
    '''  
    size_str = '_'.join([str(i) for i in size])
    res_str = '_'.join([str(i) for i in target_resolution])

    data_file_name = 'CADRADSdata_%s_size_%s_res_%s.hdf5' % (mode, size_str, res_str)

    data_file_path = os.path.join(preprocessing_folder, data_file_name)

    makefolder(preprocessing_folder)

    if not os.path.exists(data_file_path) or force_overwrite:

        logging.info('This configuration of mode, size and target resolution has not yet been preprocessed')
        logging.info('Preprocessing now!')
        prepare_data(input_folder, data_file_path, mode, size, target_resolution)

    else:

        logging.info('Already preprocessed this configuration. Loading now!')

    return h5py.File(data_file_path, 'r')


  if __name__ == '__main__':
    
    # Paths settings
    input_folder = config.data_root
    preprocessing_folder = config.preprocessing_folder
    d=load_and_maybe_process_data(input_folder, preprocessing_folder, config.data_mode, config.image_size, config.target_resolution)

import os
import glob
import numpy as np
import logging
import nibabel as nib
import gc
import h5py
from skimage import transform
from skimage import util
from skimage import measure
import cv2
from PIL import Image
import pydicom

import configuration as config

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


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
    #img_o = 2*((img_o-img_o.min())/(img_o.max()-img_o.min()))-1
    img_o = (img_o-img_o.min())/(img_o.max()-img_o.min())
    return img_o


def get_pixeldata(dicom_path):
    
    '''
    return an ndarray of the Pixel Data
    '''
    dicom_dataset = pydicom.dcmread(dicom_path)
    if 'PixelData' not in dicom_dataset:
        raise TypeError("No pixel data found in this dataset.")
    
    # Make NumPy format code, e.g. "uint16", "int32" etc
    # from two pieces of info:
    # dicom_dataset.PixelRepresentation -- 0 for unsigned, 1 for signed;
    # dicom_dataset.BitsAllocated -- 8, 16, or 32
    if dicom_dataset.BitsAllocated == 1:
        # single bits are used for representation of binary data
        format_str = 'uint8'
    elif dicom_dataset.PixelRepresentation == 0:
        format_str = 'uint{}'.format(dicom_dataset.BitsAllocated)
    elif dicom_dataset.PixelRepresentation == 1:
        format_str = 'int{}'.format(dicom_dataset.BitsAllocated)
    else:
        format_str = 'bad_pixel_representation'
    try:
        numpy_dtype = np.dtype(format_str)
    except TypeError:
        msg = ("Data type not understood by NumPy: "
               "format='{}', PixelRepresentation={}, "
               "BitsAllocated={}".format(
                   format_str,
                   dicom_dataset.PixelRepresentation,
                   dicom_dataset.BitsAllocated))
        raise TypeError(msg)
    
    pixel_array = dicom_dataset.pixel_array
    if dicom_dataset.Modality.lower().find('ct') >= 0:
        pixel_array = pixel_array * dicom_dataset.RescaleSlope + dicom_dataset.RescaleIntercept  # Obtain the CT value of the image
    pixel_array = pixel_array.astype(numpy_dtype)
    
    return pixel_array, dicom_dataset.Rows, dicom_dataset.Columns, numpy_dtype, dicom_dataset.PixelSpacing, abs(dicom_dataset.SpacingBetweenSlices)
    
    
def prepare_data(input_folder, output_file, mode, size, target_resolution):
    
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
    
    count_file = 0  # set count file default
    count_pat = 0 # set count patient default
    paths = [input_folder]  # Make stack of paths to process
    #file_addrs = []
    pat_addrs = []
    while paths:
        with os.scandir(paths.pop()) as entries:
            for entry in entries:  # loop through the folder
                if entry.name.find('PA') != -1:
                    pat_addrs.append(entry.path)
                    count_pat += 1
                #if entry.name.endswith('.dcm'):
                if not entry.is_dir():
                    #file_addrs.append(entry.path)
                    count_file += 1
                elif entry.is_dir():  #if it is a subfolder
                    # Add to paths stack to get to it eventually
                    paths.append(entry.path)
    
    if mode == '2D':
        nx, ny = size
        n_file = count_file
                   
    elif mode == '3D':
        nx, ny, nz_max = size
        n_file = count_pat
        
    else:
        raise AssertionError('Wrong mode setting. This should never happen.')
        
    # Create dataset
    
    hdf5_file.create_dataset("data", [n_file] + list(size), dtype=np.float32)
    hdf5_file.create_dataset("patient", [n_file], dtype=np.uint8)
    hdf5_file.create_dataset("class", [n_file], dtype=np.uint8)
    
    logging.info('Parsing image files')
    
    for file in range(count_pat):
        
        logging.info('----------------------------------------------------------')
        path_addr = count_pat[file]
        cad_class = path_addr.split('PA')[1]
        pat_number = path_addr.split('rads ')[1].split('\\DICOM')[0]
        logging.info('Doing patient: %s, cad rads class: %s' % (pat_number, cad_class))
        
        img = []
        
        for data in sorted(os.listdir(path_addr)):
            
            dcmPath = os.path.join(path_addr, data)
            pixel_array, x, y, numpy_dtype, PixelSpacing, SpacingBetweenSlices = get_pixeldata(dcmPath)
            
            #pre-process
            if config.standardize:
                pixel_array = standardize_image(pixel_array)
            if config.normalize:
                pixel_array = normalize_image(pixel_array)
                
            img.append(pixel_array)
        
        img = np.array(img)  # array shape [N,x,y]
        img = img.transpose([1,2,0]) # array shape [x,y,N]
        
         ### PROCESSING LOOP FOR 3D DATA ################################
            if mode == '3D':
                
                scale_vector = [PixelSpacing[0] / target_resolution[0],
                                PixelSpacing[1] / target_resolution[1],
                                SpacingBetweenSlices/ target_resolution[2]]

                img_scaled = transform.rescale(img,
                                               scale_vector,
                                               order=1,
                                               preserve_range=True,
                                               multichannel=False,
                                               mode='constant')
                
                slice_vol = np.zeros((nx, ny, nz_max), dtype=np.float32)
                
                nz_curr = img_scaled.shape[2]
        
        
        
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

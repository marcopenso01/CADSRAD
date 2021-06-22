"""
@author: Marco Penso
"""
import os
import numpy as np
import logging
import h5py
from skimage import transform
import pydicom
from sklearn.model_selection import train_test_split

import configuration as config

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
    if 'SpacingBetweenSlices' in dicom_dataset:
        z = abs(dicom_dataset.SpacingBetweenSlices)
    elif 'SliceThickness' in dicom_dataset:
        z = abs(dicom_dataset.SliceThickness)
    return pixel_array, dicom_dataset.Rows, dicom_dataset.Columns, numpy_dtype, dicom_dataset.PixelSpacing, z
    
    
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
    
    if config.split_val_train < 0 or config.split_val_train > 1:
        raise AssertionError('Inadequate number of split_val_train parameters')
    
    hdf5_file = h5py.File(output_file, "w")
                 
    #count_file = 0  # set count file default
    #count_pat = 0 # set count patient default
    paths = [input_folder]  # Make stack of paths to process
    #file_addrs = []
    pat_addrs = []
    
    while paths:
        with os.scandir(paths.pop()) as entries:
            for entry in entries:  # loop through the folder
                if entry.name.find('PA') != -1:
                    pat_addrs.append(entry.path)
                    #count_pat += 1
                #if entry.name.endswith('.dcm'):
                #if not entry.is_dir():
                    #file_addrs.append(entry.path)
                    #count_file += 1
                elif entry.is_dir():  #if it is a subfolder
                    # Add to paths stack to get to it eventually
                    paths.append(entry.path)
    
    train_addrs = []
    test_addrs = []
    
    if config.split_val_train:
        
        classes = []
        for i in range(len(pat_addrs)):
            classes.append(pat_addrs[i].split('rads ')[1].split('\\DICOM')[0])
        train_addrs, test_addrs = train_test_split(pat_addrs, test_size=config.split_val_train, stratify=classes)
        
    else:
        
        train_addrs = pat_addrs
    
    num_slices = {'test': 0, 'train': 0}
    file_list = {'test': test_addrs, 'train': train_addrs}
    
    if mode == '2D':   
        
        nx, ny = size
        for i in range(len(test_addrs)):
            num_slices['test'] += len(os.listdir(test_addrs[i]))
        for i in range(len(train_addrs)):
            num_slices['train'] += len(os.listdir(train_addrs[i]))
                   
    elif mode == '3D':
        
        nx, ny, nz_max = size
        num_slices['train'] = len(train_addrs)
        num_slices['test'] = len(test_addrs)
        
    else:
        raise AssertionError('Wrong mode setting. This should never happen.')
        
    # Create dataset
    for tt in ['train', 'test']:
        
        hdf5_file.create_dataset('data_%s' % tt, [num_slices[tt]] + list(size), dtype=np.float32)
        hdf5_file.create_dataset(str('patient_' + tt), (num_slices[tt],), dtype=np.uint8)
        hdf5_file.create_dataset(str('classe_' + tt), (num_slices[tt],), dtype=np.uint8)
    
    logging.info('Parsing image files')
    
    train_test_range = ['test', 'train'] if config.split_val_train else ['train']
    for train_test in train_test_range:
        
        count = 0
        
        for file in range(len(file_list[train_test])):

            logging.info('----------------------------------------------------------')
            path_addr = file_list[train_test][file]
            pat_number = path_addr.split('PA')[1]
            cad_class = path_addr.split('rads ')[1].split('\\DICOM')[0]
            logging.info('Doing cad class: %s, patient: %s' % (cad_class, pat_number))

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

                nz_curr = img_scaled.shape[2]

                if nz_max > 0:

                    slice_vol = np.zeros((nx, ny, nz_max), dtype=np.float32)

                    stack_from = (nz_max - nz_curr) // 2

                    if stack_from < 0:
                        raise AssertionError('nz_max is too small for the chosen through plane resolution. Consider changing'
                                             'the size or the target resolution in the through-plane.')

                    for zz in range(nz_curr):

                        slice_rescaled = img_scaled[:,:,zz]
                        slice_cropped = crop_or_pad_slice_to_size(slice_rescaled, nx, ny)
                        slice_vol[:,:,stack_from] = slice_cropped   # padding VOI (volume of interest)

                        stack_from += 1

                else:

                    slice_vol = np.zeros((nx, ny, nz_curr), dtype=np.float32)

                    for zz in range(nz_curr):

                        slice_rescaled = img_scaled[:,:,zz]
                        slice_cropped = crop_or_pad_slice_to_size(slice_rescaled, nx, ny)
                        slice_vol[:,:,zz] = slice_cropped   # padding VOI (volume of interest)

                hdf5_file[str('data_'+ train_test)][file, ...] = slice_vol[None]
                hdf5_file[str('patient_' + train_test)][file, ...] = int(pat_number)
                hdf5_file[str('classe_' + train_test)][file, ...] = int(cad_class)

            ### PROCESSING LOOP FOR 2D DATA ################################
            if mode == '2D':

                scale_vector = [PixelSpacing[0] / target_resolution[0],
                                PixelSpacing[1] / target_resolution[1]]

                for zz in range(img.shape[2]):

                    slice_img = np.squeeze(img[:, :, zz])
                    slice_rescaled = transform.rescale(slice_img,
                                                       scale_vector,
                                                       order=1,
                                                       preserve_range=True,
                                                       multichannel=False,
                                                       mode = 'constant')

                    slice_cropped = crop_or_pad_slice_to_size(slice_rescaled, nx, ny)

                    hdf5_file[str('data_'+ train_test)][count, ...] = slice_cropped[None]
                    hdf5_file[str('patient_' + train_test)][count, ...] = int(pat_number)
                    hdf5_file[str('classe_' + train_test)][count, ...] = int(cad_class)

                    count += 1    
        
    # After loop:
    hdf5_file.close()
    
    
def load_and_maybe_process_data(input_folder,
                                preprocessing_folder,
                                mode,
                                size,
                                target_resolution,
                                train,
                                force_overwrite=False):
        
    '''
    This function is used to load and if necessary preprocesses the dataset
    
    :param input_folder: Folder where the raw data is located 
    :param preprocessing_folder: Folder where the proprocessed data should be written to
    :param mode: Can either be '2D' or '3D'
    :param size: Size of the output slices/volumes in pixels/voxels
    :param target_resolution: Resolution to which the data should resampled. Should have same shape as size
    :param train: Set this to True if you want preprocess training data
    :param force_overwrite: Set this to True if you want to overwrite already preprocessed data [default: False]
     
    :return: Returns an h5py.File handle to the dataset
    '''  
    size_str = '_'.join([str(i) for i in size])
    res_str = '_'.join([str(i) for i in target_resolution])
    
    if train: 
        data_file_name = 'CADRADSdata_%s_size_%s_res_%s.hdf5' % (mode, size_str, res_str)
    else:
        data_file_name = 'testCAD_%s_size_%s_res_%s.hdf5' % (mode, size_str, res_str)
                
    data_file_path = os.path.join(preprocessing_folder, data_file_name)
    
    if not os.path.exists(preprocessing_folder):
        
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
    d=load_and_maybe_process_data(input_folder, preprocessing_folder, config.data_mode, config.image_size, config.target_resolution, train, force_overwrite)

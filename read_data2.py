"""
Created on Tue Jan 25 09:39:30 2022

@author: Marco Penso
"""

import os
import numpy as np
import h5py
import cv2
import matplotlib.pyplot as plt


def crop_or_pad_slice_to_size_specific_point(slice, nx, ny, cx, cy):
    
    if len(slice.shape) == 3:
        stack = [slice[:,:,0], slice[:,:,1], slice[:,:,2]]
        RGB = []
    else:
        stack = [slice]
        
    for i in range(len(stack)):
        img = stack[i]
        x, y = img.shape
        y1 = (cy - (ny // 2))
        y2 = (cy + (ny // 2))
        x1 = (cx - (nx // 2))
        x2 = (cx + (nx // 2))
    
        if y1 < 0:
            img = np.append(np.zeros((x, abs(y1))), img, axis=1)
            x, y = img.shape
            y1 = 0
        if x1 < 0:
            img = np.append(np.zeros((abs(x1), y)), img, axis=0)
            x, y = img.shape
            x1 = 0
        if y2 > 512:
            img = np.append(img, np.zeros((x, y2 - 512)), axis=1)
            x, y = img.shape
        if x2 > 512:
            img = np.append(img, np.zeros((x2 - 512, y)), axis=0)
    
        slice_cropped = img[x1:x1 + nx, y1:y1 + ny]
        if len(stack)>1:
            RGB.append(slice_cropped)
        
    if len(stack)>1:
        return np.dstack((RGB[0], RGB[1], RGB[2]))
    else:
        return slice_cropped
    
    
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


    
input_folder = r'F:/CADRADS/DATI'

IMG = []
CAD = []
PAZ = []
CORONARIA = []

for fold in os.listdir(input_folder):
    cad_path = os.path.join(input_folder, fold)
    
    for paz in os.listdir(cad_path):
        paz_path = os.path.join(cad_path, paz)
        
        for ramo in os.listdir(paz_path):
            
            count=0
            sort = []
            for file in os.listdir(os.path.join(paz_path, ramo)):
                if count % 2 == 0:
                    path_img = os.path.join(paz_path, ramo, file)
                    img = np.array(cv2.imread(path_img,0)).astype("uint8")
                    img = crop_or_pad_slice_to_size_specific_point(img, 610, 54, 470, 250)
                    img = img[..., np.newaxis]
                    sort.append[img]
            for i in range(len(sort)):
                if i ==0:
                    img = sort[0]
                else:
                    img = np.concatenate((img, sort[i]), axis =-1)
            CAD.append(fold.split('CAD')[-1])
            IMG.append(img)
            PAZ.append(paz.split('PAZ ')[-1])
            if ramo == 'Circonflessa':
                CONORANRIA.append(0)
            elif ramo == 'Coronaria destra':
                CONORANRIA.append(1)
            elif ramo == 'Ramo marginale':
                CONORANRIA.append(2)
            elif ramo == 'Coronaria sinistra':
                CONORANRIA.append(3)
            elif ramo == 'Discendente anteriore':
                CONORANRIA.append(4)
            elif ramo == 'Discendente posteriore':
                CONORANRIA.append(5)
            elif ramo == 'Prima diagonale':
                CONORANRIA.append(6)
            elif ramo == 'Branca intermedia':
                CONORANRIA.append(7)
            elif ramo == 'Secondo diagonale':
                CONORANRIA.append(8)
            
output_fold = os.path.join(input_folder, 'pre_proc')
if not os.path.exists(output_fold):
        makefolder(output_fold)
        
hdf5_file = h5py.File(os.path.join(output_fold, 'data.hdf5'), "w")


hdf5_file.create_dataset('paz', (len(PAZ),), dtype=np.uint8)
hdf5_file.create_dataset('cad', (len(CAD),), dtype=np.uint8)
hdf5_file.create_dataset('img', [len(IMG)] + [614, 54, 4], dtype=np.uint8)
hdf5_file.create_dataset('tratto', (len(CONORANRIA),), dtype=np.uint8)

for i in range(len(PAZ)):
     hdf5_file['paz'][i, ...] = PAZ[i]
     hdf5_file['cad'][i, ...] = CAD[i]
     hdf5_file['img'][i, ...] = IMG[i]
     hdf5_file['tratto'][i, ...] = CONORANRIA[i]
  
    # After loop:
    hdf5_file.close()

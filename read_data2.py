"""
Created on Tue Feb  8 18:15:12 2022

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
        if y2 > slice.shape[1]:
            img = np.append(img, np.zeros((x, y2 - 512)), axis=1)
            x, y = img.shape
        if x2 > slice.shape[0]:
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


def contorno(img):
    index = img[:,:,0] > 60
    a = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    a[index] = 1
    #plt.imshow(a)
    contours, hier = cv2.findContours(a, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 1:
        largestContourArea = 0
        for cnt in contours:
            contourArea = cv2.contourArea(cnt)
            if( contourArea > largestContourArea):
                largestContour = cnt
                largestContourArea = contourArea
        contours = largestContour
        
    top_left_x = 1200
    top_left_y = 1200
    bottom_right_x = 0
    bottom_right_y = 0
    for cntr in contours:
        x,y,w,h = cv2.boundingRect(cntr)
        if x < top_left_x:
            top_left_x = x
        if y < top_left_y:
            top_left_y= y
        if x+w-1 > bottom_right_x:
            bottom_right_x = x+w-1
        if y+h-1 > bottom_right_y:
            bottom_right_y = y+h-1        
    top_left = (top_left_x, top_left_y)
    bottom_right = (bottom_right_x, bottom_right_y)
    #print('top left=',top_left)
    #print('bottom right=',bottom_right)
    cx = int((top_left[1]+bottom_right[1])/2)   #row
    cy = int((top_left[0]+bottom_right[0])/2)   #column
    len_x = int(bottom_right[1]-top_left[1]) -5
    len_y = int(bottom_right[0]-top_left[0]) -5
    #print(len_x, len_y)
    return len_x, len_y, cx, cy
    

def select_roi(img):
    #funzione che serve a raffinare il crop, andando a ritagliare la coronaria.
    siz = img.shape
    
    return img[int(siz[0]/100*10)::, int(siz[1]/100*15):int(siz[1]/100*85)]
    

#step 1: crop
input_folder = r'F:/CADRADS/DATI'
output_folder = r'F:/CADRADS/PREPROC'

if not os.path.exists(output_folder):
    makefolder(output_folder)
    
cad = os.listdir(input_folder)[4]
    
out_cad = os.path.join(output_folder, cad)
if not os.path.exists(out_cad):
    makefolder(out_cad)
    
cad_path = os.path.join(input_folder, cad)
print('------------ CAD: %s ---------' % cad)

for paz in os.listdir(cad_path):
    
    out_paz = os.path.join(out_cad, paz)
    if not os.path.exists(out_paz):
        makefolder(out_paz)
    
    paz_path = os.path.join(cad_path, paz)
    print('processing paz:', paz)
    
    for ramo in os.listdir(paz_path):
        if ramo in ['Circonflessa','CX','Coronaria destra','CDX','CDx','Ramo marginale','Ramo marginale ottuso','MO','Coronaria sinistra','Discendente anteriore','IVA','Discendente posteriore','Prima diagonale','Primo ramo diagonale','I DIAG','Branca intermedia','Secondo diagonale','II DIAG','Seconda discendente anteriore','Ramo posterolaterale','Secondo ramo marginale']:
            
            out_ramo = os.path.join(out_paz, ramo)
            if not os.path.exists(out_ramo):
                makefolder(out_ramo)
                
            if os.path.exists(os.path.join(paz_path,'Riferimenti placca',ramo)):
                flag = 1
                out_ramo_rif = os.path.join(out_paz, 'Riferimenti placca')
                if not os.path.exists(out_ramo_rif):
                    makefolder(out_ramo_rif)
                out_ramo_rif = os.path.join(out_ramo_rif, ramo)
                if not os.path.exists(out_ramo_rif):
                    makefolder(out_ramo_rif)
            else:
                flag = 0
            
            if cad != 'CAD0' and flag == 0:
                raise Exception('name fold error in paz %s' % paz)
                
            if cad == 'CAD0':
                file = os.listdir(os.path.join(paz_path,ramo))[0]
                path_img = os.path.join(paz_path,ramo,file)
            else:
                file = os.listdir(os.path.join(paz_path, 'Riferimenti placca',ramo))[0]
                path_img = os.path.join(paz_path,'Riferimenti placca',ramo,file)
            
            img = np.array(cv2.imread(path_img))
            img = np.rot90(img)
            r = cv2.selectROI(img)
            print(r)
            cv2.destroyAllWindows()
            
            count=0
            for file in os.listdir(os.path.join(paz_path,ramo)):
                
                if count % 2 == 0:
                    path_img = os.path.join(paz_path,ramo,file)  
                    img = np.array(cv2.imread(path_img,0)).astype("uint8")  
                    img = np.rot90(img)
                    img = img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])] 
                    fname = os.path.join(out_ramo, file.split('.')[0]+'.png')
                    cv2.imwrite(fname, img)
                    
                    if flag:
                        path_img = os.path.join(paz_path,'Riferimenti placca',ramo,file)
                        img = np.array(cv2.imread(path_img,0)).astype("uint8")
                        img = np.rot90(img)
                        img = img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
                        fname = os.path.join(out_ramo_rif, file.split('.')[0]+'.png')
                        cv2.imwrite(fname, img)
                    
                    
                plt.figure()
                plt.imshow(img)
                plt.title(str(cad + paz))

                count +=1



input_folder = r'F:/CADRADS/PREPROC'   
for cad in os.listdir(input_folder):
    cad_path = os.path.join(input_folder, cad)
    for paz in os.listdir(cad_path):
        paz_path = os.path.join(cad_path, paz)
        for ramo in os.listdir(paz_path):
            
            if cad == 'CAD0':
                if ramo in ['Circonflessa','CX','Coronaria destra','CDX','CDx','Ramo marginale','Ramo marginale ottuso','MO','Coronaria sinistra','Discendente anteriore','IVA','Discendente posteriore','Prima diagonale','Primo ramo diagonale','I DIAG','Branca intermedia','Secondo diagonale','II DIAG','Seconda discendente anteriore','Ramo posterolaterale','Secondo ramo marginale']:
                    file = os.listdir(os.path.join(paz_path,ramo))[0]
                    path_img = os.path.join(paz_path,ramo,file)
                    img = np.array(cv2.imread(path_img,0)).astype("uint8")
                    plt.figure()
                    plt.imshow(img)
                    plt.title(str(cad + paz))
                else:
                    print(cad, paz, ramo)
                    raise TypeError('fold name error')
            else:
                if ramo in ['Circonflessa','CX','Coronaria destra','CDX','CDx','Ramo marginale','Ramo marginale ottuso','MO','Coronaria sinistra','Discendente anteriore','IVA','Discendente posteriore','Prima diagonale','Primo ramo diagonale','I DIAG','Branca intermedia','Secondo diagonale','II DIAG','Seconda discendente anteriore','Ramo posterolaterale','Secondo ramo marginale']:
                    if not os.path.exists(os.path.join(paz_path,'Riferimenti placca',ramo)):
                        print(cad, paz, ramo)
                        raise TypeError('fold name error')
                    else:
                        if len(os.listdir(os.path.join(paz_path,'Riferimenti placca',ramo))) !=4:
                            print(cad, paz, ramo)
                            raise TypeError('n file error')
                        else:
                            file = os.listdir(os.path.join(paz_path,'Riferimenti placca',ramo))[0]
                            path_img = os.path.join(paz_path,'Riferimenti placca',ramo,file)
                            img = np.array(cv2.imread(path_img,0)).astype("uint8")
                            plt.figure()
                            plt.imshow(img)
                            plt.title(str(cad + paz))
                            

                

#step 2: resize
input_folder = r'F:/CADRADS/PREPROC'
output_folder = r'F:/CADRADS/RESIZE'
nx = 350
ny = 30
if not os.path.exists(output_folder):
    makefolder(output_folder)    
for cad in os.listdir(input_folder):
    out_fold = os.path.join(output_folder, cad)
    if not os.path.exists(out_fold):
        makefolder(out_fold)
    cad_path = os.path.join(input_folder, cad)
    print('------------ CAD: %s ---------' % cad)
    for paz in os.listdir(cad_path):    
        out_paz = os.path.join(out_fold, paz)
        if not os.path.exists(out_paz):
            makefolder(out_paz)    
        paz_path = os.path.join(cad_path, paz)
        print('processing paz:', paz)
        for ramo in os.listdir(paz_path):
            if ramo in ['Circonflessa','CX','Coronaria destra','CDX','CDx','Ramo marginale','Ramo marginale ottuso','MO','Coronaria sinistra','Discendente anteriore','IVA','Discendente posteriore','Prima diagonale','Primo ramo diagonale','I DIAG','Branca intermedia','Secondo diagonale','II DIAG','Seconda discendente anteriore','Ramo posterolaterale','Secondo ramo marginale']:
                out_ramo = os.path.join(out_paz, ramo)
                if not os.path.exists(out_ramo):
                    makefolder(out_ramo)
                    
                if cad != 'CAD0':
                    flag = 1
                    out_ramo_rif = os.path.join(out_paz, 'Riferimenti placca')
                    if not os.path.exists(out_ramo_rif):
                        makefolder(out_ramo)
                    out_ramo_rif = os.path.join(out_ramo_rif, ramo)
                    if not os.path.exists(out_ramo_rif):
                        makefolder(out_ramo_rif)
                else:
                    flag = 0
                    
                count = 0
                for file in os.listdir(os.path.join(paz_path, ramo)):
                    path_img = os.path.join(paz_path, ramo, file)
                    img = np.array(cv2.imread(path_img,0)).astype("uint8")
                    img_res =  cv2.resize(img, (nx, ny), interpolation=cv2.INTER_AREA)
                    fname = os.path.join(out_ramo, file.split('.')[0]+'.png')
                    cv2.imwrite(fname, img_res)
                    
                    if flag:
                        path_img_rif = os.path.join(paz_path, 'Riferimenti placca', ramo, file)
                        img = np.array(cv2.imread(path_img_rif,0)).astype("uint8")
                        img_res =  cv2.resize(img, (nx, ny), interpolation=cv2.INTER_AREA)
                        fname = os.path.join(out_ramo_rif, file.split('.')[0]+'.png')
                        cv2.imwrite(fname, img_res)

                    if count == 0:
                        fig = plt.figure()
                        ax1 = fig.add_subplot(211)
                        ax1.set_axis_off()
                        ax1.imshow(img)
                        ax1.title.set_text(str(cad + paz))
                        ax2 = fig.add_subplot(212)
                        ax2.set_axis_off()
                        ax2.imshow(img_res)
                        ax2.title.set_text(str(ramo))
                        plt.show()
                            
                    count +=1


# dataset
input_folder = r'F:/CADRADS/RESIZE'
output_fold = r'F:/CADRADS/hdf5'
if not os.path.exists(output_fold):
    makefolder(output_fold)
    
IMG = []
CAD = []
PAZ = []
CORONARIA = []

for cad in os.listdir(input_folder):        
    cad_path = os.path.join(input_folder, cad)    
    for paz in os.listdir(cad_path):        
        paz_path = os.path.join(cad_path, paz)        
        for ramo in os.listdir(paz_path):
            if ramo in ['Circonflessa','CX','Coronaria destra','CDX','CDx','Ramo marginale','Ramo marginale ottuso','MO','Coronaria sinistra','Discendente anteriore','IVA','Discendente posteriore','Prima diagonale','Primo ramo diagonale','I DIAG','Branca intermedia','Secondo diagonale','II DIAG','Seconda discendente anteriore','Ramo posterolaterale','Secondo ramo marginale']:
                sort = []
                for file in os.listdir(os.path.join(paz_path, ramo)):
                    path_img = os.path.join(paz_path, ramo, file)                    
                    img = np.array(cv2.imread(path_img,0)).astype("uint8")
                    img = img[..., np.newaxis]
                    sort.append(img)
                for i in range(len(sort)):
                    if i ==0:
                        img = sort[0]
                    else:
                        img = np.concatenate((img, sort[i]), axis =-1)
                CAD.append(cad.split('CAD')[-1])
                IMG.append(img)
                PAZ.append(paz)
                if ramo == 'Circonflessa' or ramo == 'CX':
                    CORONARIA.append(0)
                elif ramo == 'Coronaria destra' or ramo == 'CDX' or ramo == 'CDx':
                    CORONARIA.append(1)
                elif ramo == 'Ramo marginale' or ramo == 'Ramo marginale ottuso' or ramo == 'MO':
                    CORONARIA.append(2)
                elif ramo == 'Coronaria sinistra':
                    CORONARIA.append(3)
                elif ramo == 'Discendente anteriore' or ramo == 'IVA':
                    CORONARIA.append(4)
                elif ramo == 'Discendente posteriore':
                    CORONARIA.append(5)
                elif ramo == 'Prima diagonale' or ramo == 'Primo ramo diagonale' or ramo == 'I DIAG':
                    CORONARIA.append(6)
                elif ramo == 'Branca intermedia':
                    CORONARIA.append(7)
                elif ramo == 'Secondo diagonale' or ramo == 'II DIAG':
                    CORONARIA.append(8)
                elif ramo == 'Seconda discendente anteriore':
                    CORONARIA.append(9)
                elif ramo == 'Ramo posterolaterale':
                    CORONARIA.append(10)
                elif ramo == 'Secondo ramo marginale':
                    CORONARIA.append(11)
                else:
                    raise TypeError('name error!!!')
            else:
                continue
        
hdf5_file = h5py.File(os.path.join(output_fold, 'data.hdf5'), "w")

hdf5_file.create_dataset('paz', (len(PAZ),), dtype=np.uint8)
hdf5_file.create_dataset('cad', (len(CAD),), dtype=np.uint8)
hdf5_file.create_dataset('img', [len(IMG)] + [30,350,4], dtype=np.uint8)
hdf5_file.create_dataset('ramo', (len(CORONARIA),), dtype=np.uint8)

if len(PAZ) != len(CAD) or len(PAZ) != len(IMG) or len(PAZ) != len(CORONARIA):
    raise TypeError('lunghezza datasets non uguale')

for i in range(len(PAZ)):
     hdf5_file['paz'][i, ...] = int(PAZ[i])
     hdf5_file['cad'][i, ...] = int(CAD[i])
     hdf5_file['img'][i, ...] = IMG[i]
     hdf5_file['ramo'][i, ...] = int(CORONARIA[i])
# After loop:
hdf5_file.close()


'''
path = r'F:/CADRADS/DATI/CAD3/1/Circonflessa'
path_img = os.path.join(path, os.listdir(path)[0])
print(path_img)

img = np.array(cv2.imread(path_img,0)).astype("uint8")
img2 = crop_or_pad_slice_to_size_specific_point(img, 818, 132, 440, 248)
out_path = r'F:/CADRADS/PREPROC/CAD3/1/Circonflessa'
fname = os.path.join(out_path, 'se000.png')
cv2.imwrite(fname, img2)
'''

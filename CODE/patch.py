import os
import numpy as np
import h5py
import cv2
import matplotlib.pyplot as plt

X = []
Y = []
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # print(x,",",y)
        X.append(y)
        Y.append(x)
        cv2.destroyAllWindows()
        
        
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
    

def select_roi(img):
    #funzione che serve a raffinare il crop, andando a ritagliare la coronaria.
    siz = img.shape
    
    return img[int(siz[0]/100*10)::, int(siz[1]/100*15):int(siz[1]/100*85)]



path = r'F:/CADRADS/Nuova cartella/DATI/CAD0'
output_folder = r'F:/CADRADS/Nuova cartella/PATCH1'
if not os.path.exists(output_folder):
    makefolder(output_folder)
out_path = os.path.join(output_folder, path.split('/')[-1])
if not os.path.exists(out_path):
    makefolder(out_path)

for paz in os.listdir(path):
    path_paz = os.path.join(path, paz)
    out_paz= os.path.join(out_path, paz)
    if not os.path.exists(out_paz):
        makefolder(out_paz)
    
    for ramo in os.listdir(path_paz):
        
        if ramo in ['Circonflessa','CX','Coronaria destra','CDX','CDx','Ramo marginale','Ramo marginale ottuso','MO','Coronaria sinistra','Discendente anteriore','IVA','Discendente posteriore','Prima diagonale','Primo ramo diagonale','I DIAG','Branca intermedia','Secondo diagonale','II DIAG','Seconda discendente anteriore','Ramo posterolaterale','Secondo ramo marginale']:
            print('paz:',paz, 'ramo',ramo)
            if os.path.exists(os.path.join(path_paz, ramo + str(2))):
                flag = 1
            else:
                flag = 0
            print('flag', flag)
            
            if os.path.exists(os.path.join(path_paz, 'Riferimenti placca', ramo)):
                file = os.listdir(os.path.join(path_paz, 'Riferimenti placca', ramo))[0]
                img = np.array(cv2.imread(os.path.join(path_paz,'Riferimenti placca',ramo,file)))
            else:
                file = os.listdir(os.path.join(path_paz, ramo))[0]
                img = np.array(cv2.imread(os.path.join(path_paz,ramo,file)))
            img = np.rot90(img)
            X=[]
            Y=[]
            while True:
                cv2.imshow("image", img.astype('uint8'))
                cv2.namedWindow('image')
                cv2.setMouseCallback("image", click_event)
                k = cv2.waitKey(0)
                # press 'q' to exit
                if k == ord('q') or k == 27:
                    break
                else:
                    cv2.destroyAllWindows()          
            cv2.destroyAllWindows()
            
            for ii in range(len(X)):
                if not os.path.exists(os.path.join(out_paz, ramo+'_'+str(ii))):
                    makefolder(os.path.join(out_paz, ramo+'_'+str(ii)))
                count=0
                for file in os.listdir(os.path.join(path_paz,ramo)):
                    if count % 2 == 0:
                        path_img = os.path.join(path_paz, ramo, file)
                        img = np.array(cv2.imread(path_img,0)).astype("uint8")
                        img = np.rot90(img)
                        img = crop_or_pad_slice_to_size_specific_point(img, 50, 120, X[ii], Y[ii]) 
                        fname = os.path.join(out_paz,ramo+'_'+str(ii),file.split('.')[0]+'.png')
                        cv2.imwrite(fname, img)
                    count +=1
                if flag:
                    if not os.path.exists(os.path.join(out_paz, ramo+str(2)+'_'+str(ii))):
                        makefolder(os.path.join(out_paz, ramo+str(2)+'_'+str(ii)))
                    count=0
                    for file in os.listdir(os.path.join(path_paz,ramo)):
                        if count % 2 != 0:
                            path_img = os.path.join(path_paz, ramo, file)
                            img = np.array(cv2.imread(path_img,0)).astype("uint8")
                            img = np.rot90(img)
                            img = crop_or_pad_slice_to_size_specific_point(img, 50, 120, X[ii], Y[ii]) 
                            fname = os.path.join(out_paz,ramo+str(2)+'_'+str(ii),file.split('.')[0]+'.png')
                            cv2.imwrite(fname, img)
                        count +=1
                        

# dataset
input_folder = r'F:/CADRADS/Nuova cartella/PATCH1'
output_fold = r'F:/CADRADS/Nuova cartella/hdf5_3'
if not os.path.exists(output_fold):
    makefolder(output_fold)
nx = 120
ny = 50
IMG = []
CAD = []
PAZ = []
CORONARIA = []
for cad in os.listdir(input_folder):        
    cad_path = os.path.join(input_folder, cad)    
    for paz in os.listdir(cad_path):        
        paz_path = os.path.join(cad_path, paz)        
        for ramo in os.listdir(paz_path):
            sort = []
            if len(os.listdir(os.path.join(paz_path, ramo))) == 4:
                for file in os.listdir(os.path.join(paz_path, ramo)):
                    path_img = os.path.join(paz_path, ramo, file)                    
                    img = np.array(cv2.imread(path_img,0)).astype("uint8")
                    img = img[..., np.newaxis]
                    sort.append(img)
                for i in range(len(sort)):
                    if i == 0:
                        img = sort[0]
                    else:
                        img = np.concatenate((img, sort[i]), axis =-1)
                CAD.append(cad.split('CAD')[-1])
                IMG.append(img)
                PAZ.append(paz)
                ramo = ramo.split('_')[0]
                if ramo == 'Circonflessa' or ramo == 'CX' or ramo == 'Circonflessa2' or ramo == 'CX2':
                    CORONARIA.append(0)
                elif ramo == 'Coronaria destra' or ramo == 'CDX' or ramo == 'CDx' or ramo == 'Coronaria destra2' or ramo == 'CDX2' or ramo == 'CDx2':
                    CORONARIA.append(1)
                elif ramo == 'Ramo marginale' or ramo == 'Ramo marginale ottuso' or ramo == 'MO' or ramo == 'Ramo marginale2' or ramo == 'Ramo marginale ottuso2' or ramo == 'MO2':
                    CORONARIA.append(2)
                elif ramo == 'Coronaria sinistra' or ramo == 'Coronaria sinistra2':
                    CORONARIA.append(3)
                elif ramo == 'Discendente anteriore' or ramo == 'IVA' or ramo == 'Discendente anteriore2' or ramo == 'IVA2':
                    CORONARIA.append(4)
                elif ramo == 'Discendente posteriore' or ramo == 'Discendente posteriore2':
                    CORONARIA.append(5)
                elif ramo == 'Prima diagonale' or ramo == 'Primo ramo diagonale' or ramo == 'I DIAG' or ramo == 'Prima diagonale2' or ramo == 'Primo ramo diagonale2' or ramo == 'I DIAG2':
                    CORONARIA.append(6)
                elif ramo == 'Branca intermedia' or ramo == 'Branca intermedia2':
                    CORONARIA.append(7)
                elif ramo == 'Secondo diagonale' or ramo == 'II DIAG' or ramo == 'Secondo diagonale2' or ramo == 'II DIAG2':
                    CORONARIA.append(8)
                elif ramo == 'Seconda discendente anteriore' or ramo == 'Seconda discendente anteriore2':
                    CORONARIA.append(9)
                elif ramo == 'Ramo posterolaterale' or ramo == 'Ramo posterolaterale2':
                    CORONARIA.append(10)
                elif ramo == 'Secondo ramo marginale' or ramo == 'Secondo ramo marginale2':
                    CORONARIA.append(11)
                else:
                    raise TypeError('name error!!!')
            else:
                continue
hdf5_file = h5py.File(os.path.join(output_fold, 'data.hdf5'), "w")
hdf5_file.create_dataset('paz', (len(PAZ),), dtype=np.uint8)
hdf5_file.create_dataset('cad', (len(CAD),), dtype=np.uint8)
hdf5_file.create_dataset('img', [len(IMG)] + [ny,nx,4], dtype=np.uint8)
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

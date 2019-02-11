import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
from tqdm import tqdm

IMG_PX_SIZE = 80
HM_SLICES = 16

data_dir = 'C:/users/setcodestofire/documents/mygithub/brain/FinalData'
dataset  = os.listdir(data_dir)

def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

def mean(l):
    return sum(l) / len(l)

def process_data(data):

    path = data_dir +'/'+ data
    img3d = nib.load(path)
    a = np.array(img3d.get_data())

    slice_img=[]

    if "bold" not in data:
        label = [0,1]
        for i in range(a.shape[2]):
            slice_img.append(cv2.resize(np.array(a[:,:,i]),(IMG_PX_SIZE,IMG_PX_SIZE)))
    else:
        label = [1,0]
        for i in range(a.shape[2]):
            slice_img.append(cv2.resize(np.array(a[:,:,i,0]),(IMG_PX_SIZE,IMG_PX_SIZE)))


    new_slices = []

    chunk_sizes = math.ceil(len(slice_img) / HM_SLICES)
    for slice_chunk in chunks(slice_img, chunk_sizes):
        slice_chunk = list(map(mean, zip(*slice_chunk)))
        new_slices.append(slice_chunk)

    return np.array(new_slices),np.array(label)

much_data = []

for data in tqdm(dataset):
    img_data,label = process_data(data)
    for j in range(16):
        much_data.append([img_data[j],label])

np.save('muchdata-{}-{}-{}.npy'.format(IMG_PX_SIZE,IMG_PX_SIZE,HM_SLICES), much_data)

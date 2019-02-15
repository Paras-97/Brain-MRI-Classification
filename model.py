import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
from tqdm import tqdm
from tensorflow.python.framework import ops


IMG_PX_SIZE = 80

data_dir = '/home/paras/Desktop/MRI classification/FINALBRAIN-20190203T100858Z-001/FINALBRAIN/Image'
dataset  = os.listdir(data_dir)


ops.reset_default_graph()

def process_data(data):

    path = data_dir +'/'+ str(data)
    img3d = nib.load(path)
    a = np.array(img3d.get_data())

    slice_img=[]
    if(img3d.header['dim'][4]==1):
        b = cv2.resize(np.array(a[:,:,100]),(200,200))
        for i in range(a.shape[2]):
            slice_img.append(cv2.resize(np.array(a[:,:,i]),(IMG_PX_SIZE,IMG_PX_SIZE)))
    else:
        b = cv2.resize(np.array(a[:,:,12,1]),(200,200))
        for i in range(a.shape[2]):
            slice_img.append(cv2.resize(np.array(a[:,:,i,1]),(IMG_PX_SIZE,IMG_PX_SIZE)))

    return np.array(slice_img),b

img_data,imagetoshow = process_data(dataset[0])

import tflearn
LR = 0.001

from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

convnet = input_data(shape=[None,IMG_PX_SIZE,IMG_PX_SIZE,1], name='input')

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet,2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet,2)

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet,2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet,2)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax')
#convnet = regression(convnet, optimizer='adam', learning_rate=LR,loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet)

model.load('model.tflearn')

model_out=model.predict(img_data[30].reshape(-1,IMG_PX_SIZE,IMG_PX_SIZE,1))

if np.argmax(model_out[0]) == 0:
    print('BOLD MRI')
else:
    print('T1w MRI')

import matplotlib.pyplot as plt

plt.imshow(imagetoshow)
plt.show()




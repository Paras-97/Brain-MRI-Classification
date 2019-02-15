import numpy as np

IMG_PX_SIZE = 80
HM_SLICES = 16
LR = 1e-3

MODEL_NAME = 'boldvst1w-{}-{}.model.tflearn'.format(LR, '2conv')



train_data = np.load('muchdata-80-80-16.npy')

import tflearn
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
convnet = regression(convnet, optimizer='adam', learning_rate=LR,loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')

train = train_data[:-400]
test = train_data[-400:]

X = np.array([i[0] for i in train]).reshape(-1, IMG_PX_SIZE,IMG_PX_SIZE, 1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1, IMG_PX_SIZE,IMG_PX_SIZE, 1)
test_y = [i[1] for i in test]

model.fit({'input':X},{'targets': Y}, n_epoch=3, validation_set=({'input':test_x},{'targets':test_y}),
         snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

model.save('model.tflearn')

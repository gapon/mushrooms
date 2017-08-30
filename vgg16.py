import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Lambda, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import Adam, SGD, RMSprop
from keras.utils.data_utils import get_file
from keras.preprocessing import image
from keras import backend as K
K.set_image_dim_ordering('th')

FILE_PATH = 'http://files.fast.ai/models/'

vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((3,1,1))

def vgg_preprocess(x):
    x = x - vgg_mean
    return x[:, ::-1] # reverse axis rgb->bgr

def conv_block(model, layers, filters):
    for l in range(layers):
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(filters, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2,2)))
    
def dense_block(model, units, drp_rate):
    model.add(Dense(units, activation='relu'))
    model.add(Dropout(drp_rate))
    
# Define vgg16 model    
def vgg16():
    vgg = Sequential()
    vgg.add(Lambda(vgg_preprocess, input_shape=(3,224,224), output_shape=(3,224,224)))

    conv_block(vgg, 2, 64)
    conv_block(vgg, 2, 128)
    conv_block(vgg, 3, 256)
    conv_block(vgg, 3, 512)
    conv_block(vgg, 3, 512)

    vgg.add(Flatten())

    dense_block(vgg, 4096, 0.5)
    dense_block(vgg, 4096, 0.5)

    vgg.add(Dense(1000, activation = 'softmax'))

    fpath = get_file('vgg16.h5', FILE_PATH+'vgg16.h5', cache_subdir='models')
    vgg.load_weights(fpath)
    vgg.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return vgg
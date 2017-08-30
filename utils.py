import bcolz
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from keras.utils.np_utils import to_categorical
from keras.preprocessing import image

def get_batches(path, gen=image.ImageDataGenerator(), shuffle=True, 
                batch_size=64, class_mode='categorical'):
    return gen.flow_from_directory(path, target_size=(224,224), 
                class_mode=class_mode, shuffle=shuffle, batch_size=batch_size)

def save_array(fname, arr):
    c=bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()

def load_array(fname):
    return bcolz.open(fname)[:]

def plot(img):
    plt.imshow(to_plot(img))

def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')

def onehot(x):
    return to_categorical(x)
# from pyunpack import Archive
# Archive('/dataset.rar').extractall('/data')
# import patoolib
# from PIL import Image
# from random import sample
# from os.path import join
# from os import listdir
# import matplotlib.pyplot as plt
# import shutil
import numpy as np

# patoolib.extract_archive("dataset.rar")


NUM_OF_SAMPLE = 8500
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

img = load_img('./dataset224_2/image0.jpg')
img_array = img_to_array(img)

y = np.zeros((NUM_OF_SAMPLE, *img_array.shape))
for i in range(NUM_OF_SAMPLE):
    filename = './dataset224_2/image' + str(i) + '.jpg'
    img = load_img(filename)
    img_array = img_to_array(img)
    y[i] = img_array / 255 - 0.5

features = []
with open('./dataset224_2/test0.mhm', 'r') as file:
    a = '0'
    while (a[:8] != 'modifier'):
        a = file.readline()
    while (a[:8] == 'modifier'):
        s = a.split()
        features.append(s[0] + ' ' + s[1] + ' ')
        a = file.readline()
num_features = len(features)

X = np.zeros((NUM_OF_SAMPLE, num_features))
for i in range(NUM_OF_SAMPLE):
    filename = './dataset224_2/test' + str(i) + '.mhm'
    with open(filename, 'r') as file:
        while (a[:8] != 'modifier'):
            a = file.readline()
        feature_idx = 0
        while (a[:8] == 'modifier'):
            s = a.split()
            X[i][feature_idx] = float(s[-1])
            feature_idx += 1
            a = file.readline()

from keras.models import Sequential
from keras.layers import Conv2D, Dense, Activation, Reshape, UpSampling2D, BatchNormalization
from keras import optimizers
import keras

from keras.layers import LeakyReLU


def make_model():
    model = Sequential()
    model.add(Dense(256 * 4 * 4, input_dim=152, use_bias=False))
    model.add(LeakyReLU(alpha=0.02))
    model.add(Reshape((4, 4, 256)))
    model.add(UpSampling2D())
    model.add(Conv2D(256, kernel_size=4, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.02))
    model.add(UpSampling2D())
    model.add(Conv2D(256, kernel_size=4, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.02))
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=4, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.02))
    model.add(UpSampling2D())
    model.add(Conv2D(64, kernel_size=4, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.02))
    model.add(UpSampling2D())
    model.add(Conv2D(32, kernel_size=4, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.02))
    model.add(UpSampling2D())
    model.add(Conv2D(3, kernel_size=4, padding="same"))
    model.add(Activation("tanh"))

    return model


model = make_model()

# model.load_weights('model2803_1.hdf5')
INIT_LR = 1e-2
BATCH_SIZE = 128
EPOCHS = 10000
# def lr_scheduler(epoch):
#     return INIT_LR * 0.9 ** (epoch//50)
# optimizer = optimizers.SGD(INIT_LR, 0.9)
optimizer = optimizers.Adam(INIT_LR)
model.compile(loss='mae', optimizer=optimizer, metrics=['mae'])
# model.load_weights('model2803_1.hdf5')

# from keras_tqdm import TQDMNotebookCallback
checkpoint = keras.callbacks.ModelCheckpoint('model_224_2_{epoch:08d}.hdf5', period=100)

history = model.fit(
    # X_train, y_train,
    X[:-5], y[:-5],
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[checkpoint],
    validation_data=(X[-5:], y[-5:]),
    shuffle=True,
    verbose=1,
    # initial_epoch=0
)

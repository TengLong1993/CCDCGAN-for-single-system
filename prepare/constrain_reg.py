from __future__ import print_function, division

import keras
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam

from keras import backend as K

def r2_score(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )


import matplotlib.pyplot as plt

import sys

import numpy as np
import os

class constrain():
    def __init__(self):
        # Input shape
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the constrain network
        self.constrain = self.build_constrain()
        self.constrain.compile(loss='mean_squared_error',
            optimizer=optimizer,
            metrics=[r2_score])



    def build_constrain(self):

        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1024,activation='linear'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256,activation='linear'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256,activation='linear'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256,activation='linear'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(64,activation='linear'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1,activation='linear'))
        model.add(LeakyReLU(alpha=0.2))
        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, X_npy='./train_X.npy',y_npy='./train_y.npy',model_path='./model/formation_energy_reg.h5',epochs=50, batch_size=128, save_interval=50, split_ratio=0.9):

        # Load the dataset
        X=np.load(X_npy)
        print(X.shape)

        X = np.expand_dims(X, axis=3)

        # Adversarial ground truths
        y = np.load(y_npy)
        print(y.shape)

        train_idx = np.random.randint(0, X.shape[0], int(X.shape[0]*split_ratio))
        test_idx = []
        for i in range(X.shape[0]):
            if i in train_idx:
                j=0
            else:
                test_idx.append(i)
        X_train=X[train_idx]
        y_train=y[train_idx]

        test_idx=np.array(test_idx)
        X_test=X[test_idx]
        y_test=y[test_idx]

        for epoch in range(epochs):

            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]
            valid = y_train[idx]

            # Train the constrain
            d_loss_train = self.constrain.train_on_batch(imgs, valid)

            d_loss=self.constrain.test_on_batch(X_test, y_test)

            # Plot the progress
            print ("%d [test mse: %f, acc.: %.2f%%] [train mse: %f, acc.: %.2f%%]" % (epoch, d_loss[0], 100*d_loss[1], d_loss_train[0], 100*d_loss[1]))

        self.constrain.save(model_path)


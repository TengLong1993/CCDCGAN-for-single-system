from __future__ import print_function, division

from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt

import sys

import numpy as np
import os

from keras import backend as K

def min_formation_energy(y_true,y_pred):
    return K.mean(K.exp(y_pred))

class CCDCGAN():
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 200

        optimizer = Adam(0.0002, 0.5)

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        self.generator = self.build_generator()

        self.constrain = self.rebuild_constrain()

        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        self.discriminator.trainable = False

        valid = self.discriminator(img)

        self.constrain.trainable = False

        formation_energy = self.constrain(img)

        self.final_combined = Model(inputs=z,outputs=[valid, formation_energy])

        losses = ["binary_crossentropy", min_formation_energy]
        lossWeights = [ 1.0, 0.1]

        self.final_combined.compile(loss=losses, loss_weights=lossWeights, optimizer=optimizer)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(128 * 7 * 7, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((7, 7, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

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
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

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

    def rebuild_constrain(self):
        model=self.build_constrain()
        model.load_weights('./formation_energy_reg.h5')
        model.summary()
        model.name="constrain"
        return model

    def train(self, epochs, batch_size=128, save_interval=50,GAN_calculation_folder_path='./calculation/',X_train_name='train_X.npy'):

        X_train=np.load(GAN_calculation_folder_path+'train_X.npy')
        print(X_train.shape)

        X_train = np.expand_dims(X_train, axis=3)

        from keras.utils import to_categorical

        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        new_array = np.ones((batch_size,2))

        for epoch in range(1,1+epochs):

            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            g_loss = self.final_combined.train_on_batch(noise, [valid,valid])

            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f, from generator: %f, from constrain: %f] " % (epoch, d_loss[0], 100*d_loss[1], g_loss[0],g_loss[1],g_loss[2]))

            if epoch % save_interval == 0:
                if not os.path.exists(GAN_calculation_folder_path+'step_by_step_GAN_model/'):
                    os.makedirs(GAN_calculation_folder_path+'step_by_step_GAN_model/')
                self.generator.save(GAN_calculation_folder_path+'step_by_step_GAN_model/generator.h5' % epoch)
                self.discriminator.save(GAN_calculation_folder_path+'step_by_step_GAN_model/discriminator.h5' % epoch)

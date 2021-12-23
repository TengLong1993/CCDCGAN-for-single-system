from __future__ import print_function, division

from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam

import matplotlib.pyplot as plt

import sys

import numpy as np
import os

class DCGAN():
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 200

        optimizer = Adam(0.0002, 0.5)

        self.discriminator = self.rebuild_discriminator()

        # Build the generator
        self.generator = self.rebuild_generator()

    def rebuild_generator(self):
        model = load_model(previous_model_path+'step_by_step_GAN_model/generator.h5')
        model.summary()

        return model#Model(noise, img)

    def rebuild_discriminator(self):
        model = load_model(previous_model_path+'step_by_step_GAN_model/discriminator.h5')
        model.summary()

        return model#Model(img, validity)

    def predict(self, epochs, GAN_calculation_folder_path='./calculation/'):
        for epoch in range(1,1+epochs):

            noise = np.random.normal(0, 1, (1, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            if not os.path.exists(GAN_calculation_folder_path+'generated_2d_graph_square'):
                os.makedirs(GAN_calculation_folder_path+'generated_2d_graph_square')
            np.save(GAN_calculation_folder_path+'generated_2d_graph_square/epoch_%d.npy' % epoch,gen_imgs)



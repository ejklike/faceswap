# Based on the original https://www.reddit.com/r/deepfakes/ code sample + contribs

from keras.models import Model as KerasModel
from keras.layers import Input, Dense, Flatten, Reshape, add
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam

from .AutoEncoder import MultiAutoEncoder
from lib.pixel_shuffler import PixelShuffler

from keras.utils import multi_gpu_model

# for saving images
import os
import cv2
import numpy as np
from lib.training_data import stack_images

IMAGE_SHAPE = (64, 64, 3)
ENCODER_DIM = 1024

class Model(MultiAutoEncoder):
    def initModel(self):
        optimizer = Adam(lr=5e-5, beta_1=0.5, beta_2=0.999)
        x = Input(shape=IMAGE_SHAPE)

        self.autoencoder_dict = dict()
        for dec_name, decoder in self.decoder_dict.items():
            autoencoder = KerasModel(x, decoder(self.encoder(x)))
            if self.gpus > 1:
                self.autoencoder_dict[dec_name] = multi_gpu_model(autoencoder, self.gpus)
            else:
                self.autoencoder_dict[dec_name] = autoencoder

            self.autoencoder_dict[dec_name].compile(optimizer=optimizer, 
                                                    loss='mean_absolute_error')

    def converter(self, target):
        autoencoder = self.autoencoder_dict[target]
        return lambda img: autoencoder.predict(img)

    def conv(self, filters):
        def block(x):
            x = Conv2D(filters, kernel_size=5, strides=2, padding='same')(x)
            x = LeakyReLU(0.1)(x)
            return x
        return block

    def upscale(self, filters):
        def block(x):
            x = Conv2D(filters * 4, kernel_size=3, padding='same')(x)
            x = LeakyReLU(0.1)(x)
            x = PixelShuffler()(x)
            return x
        return block

    def res_block(self, f):
        def block(x):
            input_tensor = x
            x = Conv2D(f, kernel_size=3, use_bias=False, padding="same")(x)
            x = LeakyReLU(alpha=0.2)(x)
            x = Conv2D(f, kernel_size=3, use_bias=False, padding="same")(x)
            x = add([x, input_tensor])
            x = LeakyReLU(alpha=0.2)(x)
            return x
        return block

    def Encoder(self):
        input_ = Input(shape=IMAGE_SHAPE)
        x = input_
        x = self.conv(128)(x)
        x = self.conv(256)(x)
        x = self.conv(512)(x)
        x = self.conv(1024)(x)
        x = Dense(ENCODER_DIM)(Flatten()(x))
        x = Dense(4 * 4 * 1024)(x)
        x = Reshape((4, 4, 1024))(x)
        x = self.upscale(512)(x)
        return KerasModel(input_, x)

    def Decoder(self):
        input_ = Input(shape=(8, 8, 512))
        x = input_
        x = self.upscale(256)(x)
        x = self.upscale(128)(x)
        x = self.upscale(64)(x)
        # x = self.res_block(64)(x)
        # x = self.res_block(64)(x)
        x = Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')(x)
        return KerasModel(input_, x)
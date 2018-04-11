# Based on the https://github.com/shaoanlu/faceswap-GAN repo (master/temp/faceswap_GAN_keras.ipynb)

from keras.models import Model
from keras.layers import *
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import relu
from keras.initializers import RandomNormal
from keras.applications import *
from keras.optimizers import Adam

from lib.pixel_shuffler import PixelShuffler

from keras.utils import multi_gpu_model

netGAH5 = 'netGA_GAN.h5'
netGBH5 = 'netGB_GAN.h5'
netDAH5 = 'netDA_GAN.h5'
netDBH5 = 'netDB_GAN.h5'

conv_init = RandomNormal(0, 0.02)

class GANModel():
    img_size = 64
    channels = 3
    img_shape = (img_size, img_size, channels)
    encoded_dim = 1024
    nc_in = 3 # number of input channels of generators
    nc_D_inp = 6 # number of input channels of discriminators
    optimizer = Adam(1e-4, 0.5)

    def __init__(self, model_dir, gpus):
        self.model_dir = model_dir
        self.gpus = gpus

        # Build and compile the discriminator
        self.netDA, self.netDB = self.build_discriminator()

        # Build and compile the generator
        self.netGA, self.netGB = self.build_generator()

    def converter(self, target='B'):
        predictor = self.netGB if target == 'B' else self.netGA
        return lambda img: predictor.predict(img)

    def build_generator(self):

        def conv_block(input_tensor, f):
            x = input_tensor
            x = Conv2D(f, kernel_size=3, strides=2, kernel_initializer=conv_init, use_bias=False, padding="same")(x)
            x = Activation("relu")(x)
            return x

        def res_block(input_tensor, f):
            x = input_tensor
            x = Conv2D(f, kernel_size=3, kernel_initializer=conv_init, use_bias=False, padding="same")(x)
            x = LeakyReLU(alpha=0.2)(x)
            x = Conv2D(f, kernel_size=3, kernel_initializer=conv_init, use_bias=False, padding="same")(x)
            x = add([x, input_tensor])
            x = LeakyReLU(alpha=0.2)(x)
            return x

        def upscale_ps(filters, use_instance_norm=True):
            def block(x):
                x = Conv2D(filters*4, kernel_size=3, use_bias=False, kernel_initializer=RandomNormal(0, 0.02), padding='same')(x)
                x = LeakyReLU(0.1)(x)
                x = PixelShuffler()(x)
                return x
            return block

        def Encoder(nc_in=3, input_size=64):
            inp = Input(shape=self.img_shape)
            x = Conv2D(64, kernel_size=5, kernel_initializer=conv_init, use_bias=False, padding="same")(inp)
            x = conv_block(x,128)
            x = conv_block(x,256)
            x = conv_block(x,512)
            x = conv_block(x,1024)
            x = Dense(1024)(Flatten()(x))
            x = Dense(4 * 4 * 1024)(x)
            x = Reshape((4, 4, 1024))(x)
            out = upscale_ps(512)(x)
            return Model(inputs=inp, outputs=out)

        def Decoder_ps(nc_in=512, input_size=8):
            input_ = Input(shape=(input_size, input_size, nc_in))
            x = input_
            x = upscale_ps(256)(x)
            x = upscale_ps(128)(x)
            x = upscale_ps(64)(x)
            x = res_block(x, 64)
            x = res_block(x, 64)
            #x = Conv2D(4, kernel_size=5, padding='same')(x)
            alpha = Conv2D(1, kernel_size=5, padding='same', activation="sigmoid")(x)
            rgb = Conv2D(3, kernel_size=5, padding='same', activation="tanh")(x)
            out = concatenate([alpha, rgb])
            return Model(input_, out )

        encoder = Encoder()
        decoder_A = Decoder_ps()
        decoder_B = Decoder_ps()
        x = Input(shape=self.img_shape)
        self.netGA = Model(x, decoder_A(encoder(x)))
        self.netGB = Model(x, decoder_B(encoder(x)))

        try:
            self.netGA.load_weights(str(self.model_dir / netGAH5))
            self.netGB.load_weights(str(self.model_dir / netGBH5))
            print ("Generator models loaded.")
        except:
            print ("Generator weights files not found.")
            pass

        if self.gpus > 1:
            self.netGA = multi_gpu_model( self.netGA, self.gpus)
            self.netGB = multi_gpu_model( self.netGB, self.gpus)

        return netGA, netGB

    def build_discriminator(self):
        def conv_block_d(input_tensor, f, use_instance_norm=True):
            x = input_tensor
            x = Conv2D(f, kernel_size=4, strides=2, kernel_initializer=conv_init, use_bias=False, padding="same")(x)
            x = LeakyReLU(alpha=0.2)(x)
            return x

        def Discriminator(nc_in, input_size=64):
            inp = Input(shape=(input_size, input_size, nc_in))
            #x = GaussianNoise(0.05)(inp)
            x = conv_block_d(inp, 64, False)
            x = conv_block_d(x, 128, False)
            x = conv_block_d(x, 256, False)
            out = Conv2D(1, kernel_size=4, kernel_initializer=conv_init, use_bias=False, padding="same", activation="sigmoid")(x)
            return Model(inputs=[inp], outputs=out)

        netDA = Discriminator(self.nc_D_inp)
        netDB = Discriminator(self.nc_D_inp)
        try:
            netDA.load_weights(str(self.model_dir / netDAH5))
            netDB.load_weights(str(self.model_dir / netDBH5))
            print ("Discriminator models loaded.")
        except:
            print ("Discriminator weights files not found.")
            pass
        return netDA, netDB

    def load(self, target='B'):
        if target == 'A':
            print("target 'A' is not supported on GAN")
            # TODO load is done in __init__ => look how to swap if possible
        return True

    def save_weights(self):
        self.netGA.save_weights(str(self.model_dir / netGAH5))
        self.netGB.save_weights(str(self.model_dir / netGBH5))
        self.netDA.save_weights(str(self.model_dir / netDAH5))
        self.netDB.save_weights(str(self.model_dir / netDBH5))
        print ("Models saved.")

    # def save_images(self, target_A, target_B, epoch):
    #     test_A = target_A[0:14]
    #     test_B = target_B[0:14]

    #     figure_A = numpy.stack([
    #         test_A,
    #         self.autoencoder_A.predict( test_A ),
    #         self.autoencoder_B.predict( test_A ),
    #         ], axis=1 )
    #     figure_B = numpy.stack([
    #         test_B,
    #         self.autoencoder_B.predict( test_B ),
    #         self.autoencoder_A.predict( test_B ),
    #         ], axis=1 )

    #     figure = numpy.concatenate( [ figure_A, figure_B ], axis=0 )
    #     figure = figure.reshape( (4,7) + figure.shape[1:] )
    #     figure = stack_images( figure )

    #     figure = numpy.clip( figure * 255, 0, 255 ).astype('uint8')
    #     cv2.imwrite(str(self.model_dir / '{}.png'.format(epoch)), figure)
    #     print('saved model images')
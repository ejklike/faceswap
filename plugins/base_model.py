# Based on the https://github.com/shaoanlu/faceswap-GAN repo (master/temp/faceswap_GAN_keras.ipynb)

from keras.models import Model
from keras.layers import Input, Conv2D, Activation, Dense, Flatten, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.initializers import RandomNormal
from keras.utils import multi_gpu_model

from lib.pixel_shuffler import PixelShuffler

encoderH5 = 'encoder.h5'
decoder_AH5 = 'decoder_A.h5'
decoder_BH5 = 'decoder_B.h5'
netD_AH5 = 'netD_A.h5'
netD_BH5 = 'netD_B.h5'

conv_init = RandomNormal(0, 0.02)

def conv_block(n_filter):
    def block(x):
        """
        Original:
        x = Conv2D(n_filter, kernel_size=5, strides=2, padding='same')(x)
        x = LeakyReLU(0.1)(x)
        """
        x = Conv2D(n_filter, 
                kernel_size=5, 
                strides=2, 
                kernel_initializer=conv_init, 
                use_bias=False, 
                padding="same")(x)
        x = Activation("relu")(x)
        return x
    return block

def conv_block_d(n_filter, alpha=0.2):
    def block(x):
        x = Conv2D(n_filter, 
                kernel_size=4, 
                strides=2, 
                kernel_initializer=conv_init, 
                use_bias=False, 
                padding="same")(x)
        x = LeakyReLU(alpha=alpha)(x)
        return x
    return block

def upscale_block_ps(n_filter):
    def block(x):
        x = Conv2D(n_filter * 4, 
                   kernel_size=3, 
                   kernel_initializer=conv_init,
                   use_bias=False, 
                   padding='same')(x)
        x = LeakyReLU(0.1)(x)
        x = PixelShuffler()(x)
        return x
    return block

def res_block(n_filter):
    def block(input_tensor):
        x = input_tensor
        x = Conv2D(n_filter, 
                   kernel_size=3, 
                   kernel_initializer=conv_init, 
                   use_bias=False, 
                   padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(n_filter, 
                   kernel_size=3, 
                   kernel_initializer=conv_init, 
                   use_bias=False, 
                   padding="same")(x)
        x = add([x, input_tensor])
        x = LeakyReLU(alpha=0.2)(x)
        return x
    return block


class BaseModel(object):
    img_size = 64
    channels = 3
    img_shape = (img_size, img_size, channels)
    
    enc_dim = 1024

    enc_img_size = 8
    enc_channels = 512
    enc_img_shape = (enc_img_size, enc_img_size, enc_channels)
    
    use_discriminator = False
    nc_D_inp = 3
    
    def __init__(self, model_dir, gpus):
        self.model_dir = model_dir
        self.gpus = gpus
        
        self.define_model()

    def define_model(self):
        self.encoder = self.Encoder()
        self.decoder_A = self.Decoder()
        self.decoder_B = self.Decoder()
        
        x = Input(shape=self.img_shape)
        self.netGA = Model(x, self.decoder_A(self.encoder(x)))
        self.netGB = Model(x, self.decoder_B(self.encoder(x)))

        if self.use_discriminator is True:
            self.netDA = self.Discriminator()
            self.netDB = self.Discriminator()
        else:
            self.netDA = self.netDB = None

        if self.gpus > 1:
            self.netGA = multi_gpu_model(self.netGA, self.gpus)
            self.netGB = multi_gpu_model(self.netGB, self.gpus)

    def load_weights(self, target='B'):
        (dec_A, dec_B) = (decoder_AH5, decoder_BH5) if target == 'B' else (decoder_BH5, decoder_AH5)
        (disc_A, disc_B) = (netD_AH5, netD_BH5) if target == 'B' else (netD_BH5, netD_AH5)

        try:
            self.encoder.load_weights(str(self.model_dir / encoderH5))
            self.decoder_A.load_weights(str(self.model_dir / dec_A))
            self.decoder_B.load_weights(str(self.model_dir / dec_B))
            if self.use_discriminator is True:
                self.netDA.load_weights(str(self.model_dir / disc_A))
                self.netDB.load_weights(str(self.model_dir / disc_B))
            print('loaded model weights')
            return True
        except Exception as e:
            print('Failed loading existing model weights.')
            print(e)
            return False

    def save_weights(self):
        self.encoder.save_weights(str(self.model_dir / encoderH5))
        self.decoder_A.save_weights(str(self.model_dir / decoder_AH5))
        self.decoder_B.save_weights(str(self.model_dir / decoder_BH5))
        if self.use_discriminator is True:
            self.netDA.save_weights(str(self.model_dir / netD_AH5))
            self.netDB.save_weights(str(self.model_dir / netD_BH5))
        print('saved model weights', end='\r')

    def Encoder(self):
        inp = Input(shape=self.img_shape)
        x = inp                         # (64, 64, 3)
        # x = Conv2D(64, # from GAN model
        #            kernel_size=5, 
        #            kernel_initializer=conv_init, 
        #            use_bias=False, 
        #            padding="same")(x)
        x = conv_block(128)(x)          # (32, 32, 128)
        x = conv_block(256)(x)          # (16, 16, 256)
        x = conv_block(512)(x)          # (8, 8, 512)
        x = conv_block(1024)(x)         # (4, 4, 1024)
        x = Dense(1024)(Flatten()(x))   # (1024)
        x = Dense(4 * 4 * 1024)(x)      # (1024 * 16 = 16384)
        x = Reshape((4, 4, 1024))(x)    # (4, 4, 1024)
        out = upscale_block_ps(512)(x)  # (8, 8, 512)
        return Model(inputs=inp, outputs=out)

    def Decoder(self):
        inp = Input(shape=self.enc_img_shape)
        x = inp                         # (8, 8, 512)
        x = upscale_block_ps(256)(x)    # (16, 16, 256)
        x = upscale_block_ps(128)(x)    # (32, 32, 128)
        x = upscale_block_ps(64)(x)     # (64, 64, 64)
        # x = res_block(64)(x) # from GAN model
        # x = res_block(64)(x) # from GAN model
        out = Conv2D(3, 
                   kernel_size=5, 
                   padding='same', 
                   activation='tanh')(x) # (64, 64, 3)
        return Model(inp, out)

    def Discriminator(self):
        inp = Input(shape=self.img_shape)
        x = inp                     # (64, 64, 3)
        x = conv_block_d(64)(x)     # (32, 32, 64)
        x = conv_block_d(128)(x)    # (16, 16, 128)
        x = conv_block_d(256)(x)    # (8, 8, 256)
        x = conv_block_d(512)(x)    # (4, 4, 512)
        out = Conv2D(1,             # (1, 1, 1)
                     kernel_size=4,
                     kernel_initializer=conv_init, 
                     use_bias=False, 
                     padding="valid", 
                     activation="sigmoid")(x)   
        print(out.shape)
        return Model(inputs=[inp], outputs=out)

    def converter(self, target='B'):
        predictor = self.netGB if target == 'B' else self.netGA
        return lambda img: predictor.predict(img)
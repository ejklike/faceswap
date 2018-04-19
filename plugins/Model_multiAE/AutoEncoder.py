# AutoEncoder base classes
import cv2
import numpy as np
from lib.training_data import stack_images

encoderH5 = 'encoder.h5'
decoderH5 = 'decoder_{}.h5'

class MultiAutoEncoder(object):
    def __init__(self, model_dir, target_names, gpus):
        self.model_dir = model_dir
        self.gpus = gpus

        self.encoder = self.Encoder()
        self.decoder_dict = dict()
        for target_name in target_names:
            self.decoder_dict[target_name] = self.Decoder()

        self.initModel()

    def save_weights(self):
        self.encoder.save_weights(str(self.model_dir / encoderH5))
        for dec_name, decoder in self.decoder_dict.items():
            decoder.save_weights(str(self.model_dir / decoderH5.format(dec_name)))
        print('saved model weights')

    def load(self):
        try:
            self.encoder.load_weights(str(self.model_dir / encoderH5))
            for dec_name, decoder in self.decoder_dict.items():
                decoder.load_weights(str(self.model_dir / decoderH5.format(dec_name)))
            print('loaded model weights')
            return True
        except Exception as e:
            print('Failed loading existing model weights.')
            print(e)
            return False


    def save_images(self, target_img, target_name,  epoch):
        test = target_img[:8]

        figure = np.stack([
            test,
            self.autoencoder_dict[target_name].predict(test),
            ], axis=1 )
        # print(figure, figure.shape)
        figure = figure.reshape( (4, 2) + figure.shape[1:] )
        figure = stack_images( figure )

        figure = np.clip( figure * 255, 0, 255 ).astype('uint8')
        cv2.imwrite(str(self.model_dir / '{}_{}.png'.format(target_name, epoch)), figure)
        print('\rsaved model images', end='')
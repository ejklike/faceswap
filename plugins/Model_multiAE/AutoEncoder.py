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

    def load(self, target=None):
        # TODO: target의 decoder를 가져오는 load와, 초반 weight를 불러오는 load를 나눠야 하나..?
        if target is not None:
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


    def save_images(self, target_A, target_B, epoch):
        # TODO: target????????
        test_A = target_A[0:14]
        test_B = target_B[0:14]

        figure_A = np.stack([
            test_A,
            self.autoencoder_A.predict( test_A ),
            self.autoencoder_B.predict( test_A ),
            ], axis=1 )
        figure_B = np.stack([
            test_B,
            self.autoencoder_B.predict( test_B ),
            self.autoencoder_A.predict( test_B ),
            ], axis=1 )

        figure = np.concatenate( [ figure_A, figure_B ], axis=0 )
        figure = figure.reshape( (4,7) + figure.shape[1:] )
        figure = stack_images( figure )

        figure = np.clip( figure * 255, 0, 255 ).astype('uint8')
        cv2.imwrite(str(self.model_dir / '{}.png'.format(epoch)), figure)
        print('saved model images')
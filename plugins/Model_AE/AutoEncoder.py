# AutoEncoder base classes

encoderH5 = 'encoder.h5'
decoder_AH5 = 'decoder_A.h5'
decoder_BH5 = 'decoder_B.h5'

class AutoEncoder(object):
    def __init__(self, model_dir, gpus):
        self.model_dir = model_dir
        self.gpus = gpus

        self.encoder = self.Encoder()
        self.decoder_A = self.Decoder()
        self.decoder_B = self.Decoder()

        self.initModel()

    def load(self, target):
        (face_A, face_B) = (decoder_AH5, decoder_BH5) if target == 'B' else (decoder_BH5, decoder_AH5)

        try:
            self.encoder.load_weights(str(self.model_dir / encoderH5))
            self.decoder_A.load_weights(str(self.model_dir / face_A))
            self.decoder_B.load_weights(str(self.model_dir / face_B))
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
        print('saved model weights')

    def save_images(self, target_A, target_B, epoch):
        test_A = target_A[0:14]
        test_B = target_B[0:14]

        figure_A = numpy.stack([
            test_A,
            self.autoencoder_A.predict( test_A ),
            self.autoencoder_B.predict( test_A ),
            ], axis=1 )
        figure_B = numpy.stack([
            test_B,
            self.autoencoder_B.predict( test_B ),
            self.autoencoder_A.predict( test_B ),
            ], axis=1 )

        figure = numpy.concatenate( [ figure_A, figure_B ], axis=0 )
        figure = figure.reshape( (4,7) + figure.shape[1:] )
        figure = stack_images( figure )

        figure = numpy.clip( figure * 255, 0, 255 ).astype('uint8')
        cv2.imwrite(str(self.model_dir / '{}.png'.format(epoch)), figure)
        print('saved model images')
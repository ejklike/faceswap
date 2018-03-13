import os
import cv2
import numpy

from utils import get_image_paths, load_images, stack_images
from training_data import get_training_data

from model import autoencoder_A
from model import autoencoder_B
from model import encoder, decoder_A, decoder_B

batch_size = 64
period = 100

MODEL_DIR = "../face-model/model_180313"    # h5 files of model
IMAGE_A_DIR = "../face-data/no"             # face A
IMAGE_B_DIR = "../face-data/kim"            # face B
OUTPUT_DIR = "../face-output/output_180313" # progress visualization

ENCODER_FNAME = os.path.join(MODEL_DIR, "encoder.h5")
DECODER_A_FNAME = os.path.join(MODEL_DIR, "decoder_A.h5")
DECODER_B_FNAME = os.path.join(MODEL_DIR, "decoder_B.h5")

def save_model_weights():
    encoder  .save_weights(ENCODER_FNAME)
    decoder_A.save_weights(DECODER_A_FNAME)
    decoder_B.save_weights(DECODER_B_FNAME)
    print(" ...save model weights")

def maybe_exist(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print('make directory:', dir_name)

maybe_exist(MODEL_DIR)
maybe_exist(OUTPUT_DIR)

try:
    print('try ...')
    encoder  .load_weights(ENCODER_FNAME)
    decoder_A.load_weights(DECODER_A_FNAME)
    decoder_B.load_weights(DECODER_B_FNAME)
    print('loading weight complete')
except:
    print('except ...')
    pass

images_A = get_image_paths(IMAGE_A_DIR)
images_B = get_image_paths(IMAGE_B_DIR)

images_A = load_images( images_A ) / 255.0
images_B = load_images( images_B ) / 255.0

images_A += images_B.mean( axis=(0,1,2) ) - images_A.mean( axis=(0,1,2) )

print("Start training...")

for epoch in range(1000000):
    warped_A, target_A = get_training_data( images_A, batch_size )
    warped_B, target_B = get_training_data( images_B, batch_size )

    loss_A = autoencoder_A.train_on_batch( warped_A, target_A )
    loss_B = autoencoder_B.train_on_batch( warped_B, target_B )
    print('\r', epoch, loss_A, loss_B, end='', flush=True)

    if epoch % period == 0:
        save_model_weights()
        test_A = target_A[0:14]
        test_B = target_B[0:14]

        figure_A = numpy.stack([
            test_A,
            autoencoder_A.predict( test_A ),
            autoencoder_B.predict( test_A ),
            ], axis=1 )
        figure_B = numpy.stack([
            test_B,
            autoencoder_B.predict( test_B ),
            autoencoder_A.predict( test_B ),
            ], axis=1 )

        figure = numpy.concatenate( [ figure_A, figure_B ], axis=0 )
        figure = figure.reshape( (4,7) + figure.shape[1:] )
        figure = stack_images( figure )

        figure = numpy.clip( figure * 255, 0, 255 ).astype('uint8')
        cv2.imwrite(os.path.join(OUTPUT_DIR, str(epoch) + '.png'), figure)
import os
import argparse
import cv2
import numpy

from keras.optimizers import Adam
from keras.utils import multi_gpu_model

from utils import get_image_paths, load_images, stack_images
from preprocessing.training_data import get_training_data

from model.basic import autoencoder_A
from model.basic import autoencoder_B
from model.basic import encoder, decoder_A, decoder_B


ENCODER_FNAME = None
DECODER_A_FNAME = None
DECODER_B_FNAME = None


def maybe_exist(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print('make directory:', dir_name)


def get_images(input_A_dir, input_B_dir):
    images_A = get_image_paths(input_A_dir)
    images_B = get_image_paths(input_B_dir)

    images_A = load_images( images_A ) / 255.0
    images_B = load_images( images_B ) / 255.0

    images_A += images_B.mean( axis=(0,1,2) ) - images_A.mean( axis=(0,1,2) )
    return images_A, images_B


def save_model_weights():
    encoder  .save_weights(ENCODER_FNAME)
    decoder_A.save_weights(DECODER_A_FNAME)
    decoder_B.save_weights(DECODER_B_FNAME)
    print(" ...save model weights")


def save_model_images(target_A, target_B):
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
    cv2.imwrite(os.path.join(args.output_dir, str(epoch) + '.png'), figure)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-A', '--input-A',
                        dest="input_A",
                        default="../face-data/no",
                        help="Input directory. A directory containing training images for face A.\
                            Defaults to 'input'")
    parser.add_argument('-B', '--input-B',
                        dest="input_B",
                        default="../face-data/kim",
                        help="Input directory. A directory containing training images for face B.\
                            Defaults to 'input'")
    parser.add_argument('-m', '--model-dir',
                        dest="model_dir",
                        default="../face-model/tmp",
                        help="Model directory. This is where the training data will be stored.")
    parser.add_argument('-o', '--output-dir',
                        dest="output_dir",
                        default="../face-output/tmp",
                        help="Output directory. This is where the visualization images will be stored.")
    parser.add_argument('-b', '--batch-size',
                        type=int, 
                        default=64)
    parser.add_argument('-p', '--period',
                        type=int, 
                        default=100)
    parser.add_argument('-g', '--num_gpu',
                        type=int, 
                        default=0)
    parser.add_argument('-l', '--learning_rate',
                        type=float, 
                        default=5e-5)

    args = parser.parse_args()

    # Get model and output paths
    maybe_exist(args.model_dir)
    maybe_exist(args.output_dir)
    ENCODER_FNAME = os.path.join(args.model_dir, "encoder.h5")
    DECODER_A_FNAME = os.path.join(args.model_dir, "decoder_A.h5")
    DECODER_B_FNAME = os.path.join(args.model_dir, "decoder_B.h5")

    # Load model
    if args.num_gpu > 0:
        autoencoder_A = multi_gpu_model(autoencoder_A, gpus=args.num_gpu)
        autoencoder_B = multi_gpu_model(autoencoder_B, gpus=args.num_gpu)
    optimizer = Adam( lr=args.learning_rate, beta_1=0.5, beta_2=0.999 )
    autoencoder_A.compile( optimizer=optimizer, loss='mean_absolute_error' )
    autoencoder_B.compile( optimizer=optimizer, loss='mean_absolute_error' )

    # Try to load prev. weights
    try:
        encoder  .load_weights(ENCODER_FNAME)
        decoder_A.load_weights(DECODER_A_FNAME)
        decoder_B.load_weights(DECODER_B_FNAME)
        print('loading weight complete')
    except:
        pass

    print("Start training...")

    # Load image
    images_A, images_B = get_images(args.input_A, args.input_B)

    # Training
    for epoch in range(1000000):
        warped_A, target_A = get_training_data( images_A, args.batch_size )
        warped_B, target_B = get_training_data( images_B, args.batch_size )

        loss_A = autoencoder_A.train_on_batch( warped_A, target_A )
        loss_B = autoencoder_B.train_on_batch( warped_B, target_B )
        print('\r', epoch, loss_A, loss_B, end='', flush=True)

        if epoch % args.period == 0:
            save_model_weights()
            save_model_images(target_A, target_B)
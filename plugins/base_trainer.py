
import time
import numpy as np

from keras.layers import Input, concatenate
from keras.optimizers import Adam
from keras import backend as K

from lib.training_data import TrainingDataGenerator, stack_images

# for saving images
import os
import cv2
from lib.training_data import stack_images

lrD = 5e-5
lrG = 5e-5
beta_1 = 0.5
beta_2 = 0.999

class BaseTrainer(object):
    random_transform_args = {
        'rotation_range': 20,
        'zoom_range': 0.1,
        'shift_range': 0.05,
        'random_flip': 0.5,
    }

    def __init__(self, model, fn_A, fn_B, batch_size, *kwargs):
        self.before_init()
        K.set_learning_phase(1) # 1: training phase

        self.batch_size = batch_size
        self.model = model

        generator = TrainingDataGenerator(self.random_transform_args, 
                                          160)
        self.train_batchA = generator.minibatchAB(fn_A, self.batch_size)
        self.train_batchB = generator.minibatchAB(fn_B, self.batch_size)

        self.avg_counter = self.errDA_sum = self.errDB_sum = self.errGA_sum = self.errGB_sum = 0

        self.define_loss()
        self.after_init()

    def before_init(self):
        pass

    def after_init(self):
        pass

    def _calculate_loss(self, netD, real, fake):
        loss_D = loss_G = 0.0

        # MAE loss (basic)
        loss_G += K.mean(K.abs(fake - real))

        return loss_D, loss_G

    def define_loss(self):
        """
        distorted_A: 
            A (batch_size, 64, 64, 3) tensor, input of generator_A (netGA).
        distorted_B: 
            A (batch_size, 64, 64, 3) tensor, input of generator_B (netGB).
        fake_A: 
            (batch_size, 64, 64, 3) tensor, output of generator_A (netGA).
        fake_B: 
            (batch_size, 64, 64, 3) tensor, output of generator_B (netGB).
        path_A: 
            A function that takes distorted_A as input and outputs fake_A.
        path_B: 
            A function that takes distorted_B as input and outputs fake_B.
        real_A: 
            A (batch_size, 64, 64, 3) tensor, 
            target images for generator_A given input distorted_A.
        real_B: 
            A (batch_size, 64, 64, 3) tensor, 
            target images for generator_B given input distorted_B.
        """

        def _cycle_vars(netG):
            distorted_input = netG.inputs[0]
            fake_output = netG.outputs[0]
            fn_generate = K.function([distorted_input], [fake_output])
            return distorted_input, fake_output, fn_generate

        # inputs and outputs
        real_A = Input(shape=self.model.img_shape)
        real_B = Input(shape=self.model.img_shape)
        distorted_A, fake_A, _ = _cycle_vars(self.model.netGA)
        distorted_B, fake_B, _ = _cycle_vars(self.model.netGB)

        # losses
        loss_DA, loss_GA = self._calculate_loss(self.model.netDA, real_A, fake_A)
        loss_DB, loss_GB = self._calculate_loss(self.model.netDB, real_B, fake_B)

        # trainable weights
        weightsGA = self.model.netGA.trainable_weights
        weightsGB = self.model.netGB.trainable_weights
        if self.model.use_discriminator is True:
            weightsDA = self.model.netDA.trainable_weights
            weightsDB = self.model.netDB.trainable_weights

        # Adam(..).get_updates(...)
        training_updates = Adam(lr=lrG, beta_1=0.5).get_updates(weightsGA,[], loss_GA)
        self.netGA_train = K.function([distorted_A, real_A], [loss_GA], training_updates)
        training_updates = Adam(lr=lrG, beta_1=0.5).get_updates(weightsGB,[], loss_GB)
        self.netGB_train = K.function([distorted_B, real_B], [loss_GB], training_updates)

        if self.model.use_discriminator is True:
            training_updates = Adam(lr=lrD, beta_1=0.5).get_updates(weightsDA,[],loss_DA)
            self.netDA_train = K.function([distorted_A, real_A], [loss_DA], training_updates)
            training_updates = Adam(lr=lrD, beta_1=0.5).get_updates(weightsDB,[],loss_DB)
            self.netDB_train = K.function([distorted_B, real_B], [loss_DB], training_updates)

    def train_one_step(self, iter, save_image=False):
        epoch, warped_A, target_A = next(self.train_batchA)
        epoch, warped_B, target_B = next(self.train_batchB)

        # Train generators for one batch
        errGA = self.netGA_train([warped_A, target_A])
        errGB = self.netGB_train([warped_B, target_B])
        self.errGA_sum += errGA[0]
        self.errGB_sum += errGB[0]

        # Train dicriminators for one batch
        if self.model.use_discriminator is True:
            errDA  = self.netDA_train([warped_A, target_A])
            errDB  = self.netDB_train([warped_B, target_B])
            self.errDA_sum += errDA[0]
            self.errDB_sum += errDB[0]

        self.avg_counter += 1

        print('[%s] [%d/%s][%d] Loss_DA: %f Loss_DB: %f Loss_GA: %f Loss_GB: %f'
              % (time.strftime("%H:%M:%S"), epoch, "num_epochs", iter, self.errDA_sum/self.avg_counter, self.errDB_sum/self.avg_counter, self.errGA_sum/self.avg_counter, self.errGB_sum/self.avg_counter),
              end='\r')

        if save_image is True:
            self.save_images(target_A, target_B, iter)

    def save_images(self, target_A, target_B, epoch):
        n_image = min(10, self.batch_size)
        test_A = target_A[:n_image]
        test_B = target_B[:n_image]

        figure_A = np.stack([
            test_A,
            self.model.netGA.predict( test_A ),
            self.model.netGB.predict( test_A ),
            ], axis=1 )
        figure_B = np.stack([
            test_B,
            self.model.netGB.predict( test_B ),
            self.model.netGA.predict( test_B ),
            ], axis=1 )

        figure = np.concatenate( [ figure_A, figure_B ], axis=0 )
        figure = figure.reshape( (4, n_image//2) + figure.shape[1:] )
        figure = stack_images( figure )

        figure = np.clip((figure + 1) * 255 / 2, 0, 255).astype('uint8')
        cv2.imwrite(str(self.model.model_dir / '{}.png'.format(epoch)), figure)
        print('saved model images', end='\r')

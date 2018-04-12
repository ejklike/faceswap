
import time
import numpy as np

from keras.layers import Input
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
                                          coverage=220, 
                                          scale=6, 
                                          zoom=1)
        self.train_batchA = generator.minibatchAB(fn_A, self.batch_size)
        self.train_batchB = generator.minibatchAB(fn_B, self.batch_size)

        self.avg_counter = self.errDA_sum = self.errDB_sum = self.errGA_sum = self.errGB_sum = 0

        self.define_loss()
        self.after_init()

    def before_init(self):
        pass

    def after_init(self):
        pass

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
        def cycle_vars(netG):
            distorted_input = netG.inputs[0]
            fake_output = netG.outputs[0]
            fn_generate = K.function([distorted_input], [fake_output])
            return distorted_input, fake_output, fn_generate

        def first_order(self, img, axis=1):
            img_nrows = img.shape[1]
            img_ncols = img.shape[2]
            if axis == 1:
                return K.abs(
                    img[:, :img_nrows - 1, :img_ncols - 1, :] \
                    - img[:, 1:, :img_ncols - 1, :])
            elif axis == 2:
                return K.abs(
                    img[:, :img_nrows - 1, :img_ncols - 1, :] \
                    - img[:, :img_nrows - 1, 1:, :])
            else:
                return None


        def get_loss(netD, real, fake):
            # MSE LOSS (for LSGAN)
            mse = lambda output, target : K.mean(K.square(output-target))

            gan_loss = lambda output, target : -K.mean(
                K.log(output + 1e-12) * target + K.log(1 - output + 1e-12) * (1 - target)
            )

            LOSS_D = LOSS_G = 0.0

            # MAE loss (basic)
            loss_G += K.mean(K.abs(fake - real))

            # GAN loss
            if self.model.use_discriminator is True:
                if self.use_lsgan is True:
                    # LSGAN LOSS
                    # https://arxiv.org/abs/1611.04076
                    d_out_real = netD(real) # positive sample
                    d_out_fake = netD(fake) # negative sample
                    loss_D_real = mse(d_out_real, K.ones_like(d_out_real))
                    loss_D_fake = mse(d_out_fake, K.zeros_like(d_out_fake))
                    loss_D += loss_D_real + loss_D_fake
                    loss_G += mse(output_fake, K.ones_like(output_fake))
                elif self.use_mixup is True:
                    # MIXUP
                    # https://arxiv.org/abs/1710.09412
                    dist = Beta(self.mixup_alpha, self.mixup_alpha)
                    lam = dist.sample()
                    # ==========
                    mixup = lam * concatenate([real, distorted]) + (1 - lam) * concatenate([fake, distorted])
                    # ==========
                    output_mixup = netD(mixup)
                    loss_D = self.loss_fn(output_mixup, lam * K.ones_like(output_mixup))
                    output_fake = netD(concatenate([fake, distorted])) # dummy
                    loss_G = .5 * self.loss_fn(output_mixup, (1 - lam) * K.ones_like(output_mixup))

                if self.edge_loss is True:
                    # Edge loss (similar with total variation loss)
                    loss_G += 1 * K.mean(K.abs(self.first_order(fake, axis=1) - self.first_order(real, axis=1)))
                    loss_G += 1 * K.mean(K.abs(self.first_order(fake, axis=2) - self.first_order(real, axis=2)))

            return loss_D, loss_G

        # inputs and outputs
        real_A = Input(shape=self.model.img_shape)
        real_B = Input(shape=self.model.img_shape)
        distorted_A, fake_A, _ = cycle_vars(self.model.netGA)
        distorted_B, fake_B, _ = cycle_vars(self.model.netGB)

        # losses
        loss_DA, loss_GA = get_loss(self.model.netDA, real_A, fake_A)
        loss_DB, loss_GB = get_loss(self.model.netDB, real_B, fake_B)

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


import time
import numpy as np

from lib.training_data import TrainingDataGenerator, stack_images

class Trainer():
    random_transform_args = {
        'rotation_range': 10,
        'zoom_range': 0.05,
        'shift_range': 0.05,
        'random_flip': 0.4,
    }

    def __init__(self, model, fn_A, fn_B, batch_size, *args):
        self.batch_size = batch_size
        self.model = model

        generator = TrainingDataGenerator(self.random_transform_args, 160)
        self.images_A = generator.minibatchAB(fn_A, self.batch_size)
        self.images_B = generator.minibatchAB(fn_B, self.batch_size)

    def train_one_step(self, iter, save_image=False):
        epoch, warped_A, target_A = next(self.images_A)
        epoch, warped_B, target_B = next(self.images_B)

        loss_A = self.model.autoencoder_A.train_on_batch(warped_A, target_A)
        loss_B = self.model.autoencoder_B.train_on_batch(warped_B, target_B)
        print("\r[{0}] [#{1:05d}] loss_A: {2:.5f}, loss_B: {3:.5f}".format(
            time.strftime("%H:%M:%S"), iter, loss_A, loss_B), end='', flush=True)
        
        if save_image is True:
            self.model.save_images(target_A, target_B, epoch)

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

    def __init__(self, model, target_image_path_dict, batch_size, *args):
        self.batch_size = batch_size
        self.model = model

        generator = TrainingDataGenerator(self.random_transform_args, 160)
        self.minibatch_dict = dict()
        for target_name, target_path in target_image_path_dict.items():
            self.minibatch_dict[target_name] = generator.minibatchAB(target_path, self.batch_size)

    def train_one_step(self, epoch, save_image=False):
        for i, (target_name, target_minibatch) in enumerate(self.minibatch_dict.items()):
            for _ in range(100):
                _, warped_img, target_img = next(target_minibatch)
                loss = self.model.autoencoder_dict[target_name].train_on_batch(warped_img, target_img)
                print("\r[{}] [#{:05d}] target: {} ({}/{}), loss: {:.5f}".format(
                     time.strftime("%H:%M:%S"), epoch, target_name, i, len(self.minibatch_dict), loss), end='', flush=True)

            if save_image is True:
                self.model.save_images(target_img, target_name, epoch)
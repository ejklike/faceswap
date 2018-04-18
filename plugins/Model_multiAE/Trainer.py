
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
        current_batch_dict = dict()
        for target_name, target_model in self.minibatch_dict.items():
            epoch, warped_img, target_img = next(self.minibatch_dict[target_name])
            current_batch_dict[target_name] = (warped_img, target_img)

        for target_name, target_model in self.model.autoencoder_dict.items():
            warped_img, target_img = current_batch_dict[target_name]
            loss = target_model.train_on_batch(warped_img, target_img)
            print("\r[{}] [#{:05d}] target: {}, loss: {:.5f}".format(
                time.strftime("%H:%M:%S"), epoch, target_name, loss), end='', flush=True)
        
        # if save_image is True:
        #     self.model.save_images(target_A, target_B, epoch)
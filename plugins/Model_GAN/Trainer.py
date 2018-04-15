from keras.layers import concatenate
from keras.losses import binary_crossentropy, mean_absolute_error, mean_squared_error
from keras import backend as K

from plugins.base_trainer import BaseTrainer

def first_order(x, axis=1):
    nrows = x.shape[1]
    ncols = x.shape[2]
    if axis == 1:
        return K.abs(x[:, :nrows - 1, :ncols - 1, :] - x[:, 1:, :ncols - 1, :])
    elif axis == 2:
        return K.abs(x[:, :nrows - 1, :ncols - 1, :] - x[:, :nrows - 1, 1:, :])
    else:
        return None

class Trainer(BaseTrainer):

    def _calculate_loss(self, netD, real, fake):
        loss_D = loss_G = 0.0

        # MAE loss (basic, for reconstruction)
        loss_G += K.mean(K.abs(fake - real))

        # # Edge loss (similar with total variation loss) 
        # loss_G += K.mean(first_order(fake, axis=1))# - first_order(real, axis=1))
        # loss_G += K.mean(first_order(fake, axis=2))# - first_order(real, axis=2))

        # Data for Discrimination
        d_out_real = netD(real) # positive sample
        d_out_fake = netD(fake) # negative sample

        # Original GAN
        # eps = 1e-12
        # loss_fn = lambda output, target : -K.mean(
        #     K.log(output + eps) * target + K.log(1 - output + eps) * (1 - target))
        # LS GAN
        loss_fn = lambda output, target : K.mean(K.abs(K.square(output - target)))

        # # Discrimination loss
        loss_D += loss_fn(d_out_real, K.ones_like(d_out_real))
        loss_D += loss_fn(d_out_fake, K.zeros_like(d_out_fake))
        # # Generative loss
        loss_G += loss_fn(d_out_fake, K.ones_like(d_out_fake))


        # Wasserstein GAN
        # 
        # loss_D += K.mean(d_out_fake) - K.mean(d_out_real)
        # loss_G -= K.mean(d_out_fake)

        return loss_D, loss_G
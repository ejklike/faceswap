from keras.layers import concatenate
from keras.losses import binary_crossentropy
from keras import backend as K

from plugins.base_trainer import BaseTrainer

class Trainer(BaseTrainer):

    def _calculate_loss(self, netD, real, fake):
        loss_D = loss_G = 0.0

        # MAE loss (basic)
        loss_G += K.mean(K.abs(fake - real))

        # Data for Discrimination
        d_out_real = netD(real) # positive sample
        d_out_fake = netD(fake) # negative sample

        d_out = concatenate([d_out_real, d_out_fake])
        label = concatenate([K.ones_like(d_out_real), K.zeros_like(d_out_fake)])
        print(d_out.shape, label.shape, d_out_real, d_out_fake) # ???

        # Minimize Cross Entropy for better discrimination
        # bce = lambda output, target : -K.mean(
        #     K.log(output + 1e-12) * target + K.log(1 - output + 1e-12) * (1 - target)
        # )
        # loss_D += bce(d_out, label)
        
        # loss_G -= K.mean(binary_crossentropy(label, d_out))
        loss_D += K.mean(binary_crossentropy(label, d_out))

        return loss_D, loss_G
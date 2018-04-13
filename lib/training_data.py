import cv2
import numpy as np
from random import shuffle

from .umeyama import umeyama

class TrainingDataGenerator():
    def __init__(self, random_transform_args, coverage=220, scale=6): #TODO thos default should stay in the warp function
        self.random_transform_args = random_transform_args
        self.coverage = coverage # 값이 작을수록 얼굴 가운데 부분을 자른다
        self.scale = scale

    def minibatchAB(self, images, batchsize):
        batch = self.minibatch(images, batchsize)
        for ep1, warped_img, target_img in batch.iterator():
            yield ep1, warped_img, target_img

    # A generator function that yields epoch, batchsize of warped_img and batchsize of target_img
    def minibatch(self, data, batchsize):
        length = len(data)
        assert length >= batchsize, "Number of images is lower than batch-size (Note that too few images may lead to bad training). # images: {}, batch-size: {}".format(length, batchsize)
        epoch = i = 0
        shuffle(data)
        while True:
            size = batchsize
            if i+size > length:
                shuffle(data)
                i = 0
                epoch+=1
            rtn = np.float32([self.read_image(img) for img in data[i:i+size]])
            i+=size
            yield epoch, rtn[:,0,:,:,:], rtn[:,1,:,:,:]       

    def read_image(self, fn):
        def _color_adjust(img):
            # return img / 255.0
            return img / 255.0 * 2 - 1
        
        def _random_transform(image, 
                              rotation_range, 
                              zoom_range, 
                              shift_range, 
                              random_flip):
            h, w, _ = image.shape
            rotation = np.random.uniform(-rotation_range, rotation_range)
            scale = np.random.uniform(1 - zoom_range, 1 + zoom_range)
            tx = np.random.uniform(-shift_range, shift_range) * w
            ty = np.random.uniform(-shift_range, shift_range) * h
            mat = cv2.getRotationMatrix2D((w // 2, h // 2), rotation, scale)
            mat[:, 2] += (tx, ty)
            result = cv2.warpAffine(
                image, mat, (w, h), borderMode=cv2.BORDER_REPLICATE)
            if np.random.random() < random_flip:
                result = result[:, ::-1]
            return result

        # get pair of random warped images from aligned face image
        def _random_warp(image, coverage, scale):
            assert image.shape == (256, 256, 3)
            num = 5
            out_size = 64

            # num x num lattice points in center region of input image
            range_ = np.linspace(128 - coverage//2, 128 + coverage//2, num=num)
            mapx = np.broadcast_to(range_, (num, num))
            mapy = mapx.T

            # random scattering
            mapx = mapx + np.random.normal(size=(num, num), scale=scale)
            mapy = mapy + np.random.normal(size=(num, num), scale=scale)

            # interp_size = 80
            # interested_region = 8:72 (size 64)
            interp_size = (out_size + 8*2)
            interp_min, interp_max = (0 + 8), (interp_size - 8)
            interp_mapx = cv2.resize(mapx, (interp_size, interp_size))[interp_min:interp_max, interp_min:interp_max].astype('float32')
            interp_mapy = cv2.resize(mapy, (interp_size, interp_size))[interp_min:interp_max, interp_min:interp_max].astype('float32')

            warped_image = cv2.remap(image, interp_mapx, interp_mapy, cv2.INTER_LINEAR)

            src_points = np.stack([mapx.ravel(), mapy.ravel()], axis=-1)
            dst_points = np.mgrid[0:65:16, 0:65:16].T.reshape(-1,2)
            mat = umeyama(src_points, dst_points, True)[0:2]

            target_image = cv2.warpAffine(image, mat, (out_size, out_size))
            
            return warped_image, target_image

        try:
            image = _color_adjust(cv2.imread(fn))
        except TypeError:
            raise Exception("Error while reading image", fn)

        image = cv2.resize(image, (256, 256))
        image = _random_transform(image, **self.random_transform_args)
        warped_img, target_img = _random_warp(image, self.coverage, self.scale)

        return warped_img, target_img


def stack_images(images):
    def get_transpose_axes(n):
        if n % 2 == 0:
            y_axes = list(range(1, n - 1, 2))
            x_axes = list(range(0, n - 1, 2))
        else:
            y_axes = list(range(0, n - 1, 2))
            x_axes = list(range(1, n - 1, 2))
        return y_axes, x_axes, [n - 1]
    
    images_shape = np.array(images.shape)
    new_axes = get_transpose_axes(len(images_shape))
    new_shape = [np.prod(images_shape[x]) for x in new_axes]
    return np.transpose(
        images,
        axes=np.concatenate(new_axes)
        ).reshape(new_shape)
# Based on: https://gist.github.com/anonymous/d3815aba83a8f79779451262599b0955 
#           found on https://www.reddit.com/r/deepfakes/

import cv2
import numpy as np

from lib.aligner import get_align_mat

def color_hist_match( src_im, tar_im, mask):
    def hist_match(source, template, mask=None):
        # Code borrowed from:
        # https://stackoverflow.com/questions/32655686/histogram-matching-of-two-images-in-python-2-x
        if mask is not None:
            masked_source = source * mask
            masked_template = template * mask
        else:
            masked_source = source
            masked_template = template

        oldshape = source.shape
        source = source.ravel()
        template = template.ravel()
        masked_source = masked_source.ravel()
        masked_template = masked_template.ravel()
        s_values, bin_idx, s_counts = np.unique(source, return_inverse=True, return_counts=True)
        t_values, t_counts = np.unique(template, return_counts=True)
        ms_values, mbin_idx, ms_counts = np.unique(source, return_inverse=True, return_counts=True)
        mt_values, mt_counts = np.unique(template, return_counts=True)

        s_quantiles = np.cumsum(s_counts).astype(np.float64)
        s_quantiles /= s_quantiles[-1]
        t_quantiles = np.cumsum(t_counts).astype(np.float64)
        t_quantiles /= t_quantiles[-1]
        interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

        return interp_t_values[bin_idx].reshape(oldshape)

    matched_R = hist_match(src_im[:,:,0], tar_im[:,:,0], mask)
    matched_G = hist_match(src_im[:,:,1], tar_im[:,:,1], mask)
    matched_B = hist_match(src_im[:,:,2], tar_im[:,:,2], mask)
    matched = np.stack((matched_R, matched_G, matched_B), axis=2).astype(src_im.dtype)
    return matched


class Convert():
    def __init__(self, encoder, trainer, 
                 blur_size=2, 
                 seamless_clone=False, 
                 mask_type="facehullandrect", 
                 erosion_kernel_size=None, 
                 match_histogram=False, 
                 **kwargs):
        self.encoder = encoder
        self.trainer = trainer
        # self.erosion_kernel = None
        self.erosion_kernel_size = erosion_kernel_size
        if erosion_kernel_size is not None:
            if erosion_kernel_size > 0:
                self.erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(erosion_kernel_size,erosion_kernel_size))
            elif erosion_kernel_size < 0:
                self.erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(abs(erosion_kernel_size),abs(erosion_kernel_size)))
        self.blur_size = blur_size
        self.seamless_clone = seamless_clone
        self.match_histogram = match_histogram
        self.mask_type = mask_type.lower() # Choose in 'FaceHullAndRect','FaceHull','Rect'

    def patch_image( self, image, face_detected, size ):

        image_size = image.shape[1], image.shape[0]

        mat = np.array(get_align_mat(face_detected)).reshape(2,3) * size

        new_face = self.get_new_face(image, mat, size)

        image_mask = self.get_image_mask( image, new_face, face_detected, mat, image_size )

        return self.apply_new_face(image, new_face, image_mask, mat, image_size, size)

    def apply_new_face(self, image, new_face, image_mask, mat, image_size, size):
        base_image = np.copy( image )
        new_image = np.copy( image )

        cv2.warpAffine( new_face, mat, image_size, new_image, cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC, cv2.BORDER_TRANSPARENT )

        outImage = None
        if self.seamless_clone:
            unitMask = np.clip( image_mask * 255, 0, 255 ).astype(np.uint8)

            maxregion = np.argwhere(unitMask==255)

            if maxregion.size > 0:
              miny,minx = maxregion.min(axis=0)[:2]
              maxy,maxx = maxregion.max(axis=0)[:2]
              lenx = maxx - minx;
              leny = maxy - miny;
              masky = int(minx+(lenx//2))
              maskx = int(miny+(leny//2))
              outimage = cv2.seamlessClone(new_image.astype(np.uint8),base_image.astype(np.uint8),unitMask,(masky,maskx) , cv2.NORMAL_CLONE )

              return outimage

        foreground = cv2.multiply(image_mask, new_image.astype(float))
        background = cv2.multiply(1.0 - image_mask, base_image.astype(float))
        outimage = cv2.add(foreground, background)

        return outimage

    def get_new_face(self, image, mat, size):
        face = cv2.warpAffine( image, mat, (size,size) )
        face = np.expand_dims( face, 0 )
        face_clipped = np.clip(face[0], 0, 255).astype( image.dtype )
        new_face = None
        mask = None

        # if "GAN" not in self.trainer:
        #     normalized_face = face / 255.0
        #     new_face = self.encoder(normalized_face)[0]
        #     new_face = np.clip( new_face * 255, 0, 255 ).astype( image.dtype )
        # else:
        #     normalized_face = face / 255.0 * 2 - 1
        #     fake_output = self.encoder(normalized_face)
        #     mask = fake_output[:,:,:, :1]
        #     new_face = fake_output[:,:,:, 1:]
        #     new_face = mask * new_face + (1 - mask) * normalized_face
        #     new_face = np.clip((new_face[0] + 1) * 255 / 2, 0, 255).astype( image.dtype )
        
        normalized_face = face / 255.0 * 2 - 1
        new_face = self.encoder(normalized_face)[0]
        new_face = np.clip( (new_face + 1) * 255 / 2, 0, 255 ).astype( image.dtype )

        # to check!
        cv2.imwrite('./output/source.png', face_clipped)
        cv2.imwrite('./output/target.png', new_face)

        if self.match_histogram:
            new_face = self.color_hist_match(new_face, face_clipped, mask)

        return new_face

    def get_image_mask(self, image, new_face, face_detected, mat, image_size):

        face_mask = np.zeros(image.shape,dtype=float)
        if 'rect' in self.mask_type:
            face_src = np.ones(new_face.shape, dtype=float)
            cv2.warpAffine( face_src, mat, image_size, face_mask, cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC, cv2.BORDER_TRANSPARENT )

        hull_mask = np.zeros(image.shape, dtype=float)
        if 'hull' in self.mask_type:
            hull = cv2.convexHull( np.array( face_detected.landmarksXY ).reshape((-1,2)).astype(int) ).flatten().reshape( (-1,2) )
            cv2.fillConvexPoly( hull_mask,hull, (1,1,1) )

        if self.mask_type == 'rect':
            image_mask = face_mask
        elif self.mask_type == 'facehull':
            image_mask = hull_mask
        else:
            image_mask = ((face_mask * hull_mask))

        # if self.erosion_kernel is not None:
        #     if self.erosion_kernel_size > 0:
        #         image_mask = cv2.erode(image_mask,
        #                                self.erosion_kernel,
        #                                iterations=self.erosion_kernel_size)
        #     elif self.erosion_kernel_size < 0:
        #         dilation_kernel = abs(self.erosion_kernel)
        #         image_mask = cv2.dilate(image_mask,
        #                                 dilation_kernel,
        #                                 iterations=-self.erosion_kernel_size)

        if self.blur_size != 0:
            image_mask = cv2.blur(image_mask, (self.blur_size, self.blur_size))

        return image_mask

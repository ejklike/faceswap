import cv2
import numpy
import os
from pathlib import Path
from scandir import scandir

image_extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff"]


def get_folder(path):
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_target_paths(target_dir='./data', 
                     model_dir='./model'):
    target_names = []
    target_paths = []

    if not os.path.exists(target_dir):
        target_dir = get_folder(target_dir).path

    target_scanned = list(scandir(target_dir))
    for target in target_scanned:
        if target.name[0] != '.':
            print('   - ' + target.name)
            target_model_dir = '{}/{}'.format(model_dir, target.name)
            print(get_folder(target_model_dir), target_model_dir)
            target_names.append(target.name)
            target_paths.append(get_image_paths(target.path))
    return target_names, target_paths


def get_image_paths(directory):
    dir_contents = []

    if not os.path.exists(directory):
        directory = get_folder(directory).path

    dir_scanned = list(scandir(directory))
    for x in dir_scanned:
        if any([x.name.lower().endswith(ext) for ext in image_extensions]):
            dir_contents.append(x.path)

    return dir_contents


def load_images( image_paths, convert=None ):
    iter_all_images = ( cv2.imread(fn) for fn in image_paths )
    if convert:
        iter_all_images = ( convert(img) for img in iter_all_images )
    for i,image in enumerate( iter_all_images ):
        if i == 0:
            all_images = numpy.empty( ( len(image_paths), ) + image.shape, dtype=image.dtype )
        all_images[i] = image
    return all_images
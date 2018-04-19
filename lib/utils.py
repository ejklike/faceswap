import cv2
import numpy
import os
from pathlib import Path
from scandir import scandir

image_extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff"]

def set_tf_allow_growth():
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = "0"
    set_session(tf.Session(config=config))


def get_folder(path):
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# def get_target_paths(target_dir='./data', 
#                      model_dir='./model'):
#     target_names = []
#     target_paths = []

#     if not os.path.exists(target_dir):
#         target_dir = get_folder(target_dir).path

#     target_scanned = list(scandir(target_dir))
#     for target in target_scanned:
#         if target.name[0] != '.':
#             print('   - ' + target.name)
#             target_model_dir = '{}/{}'.format(model_dir, target.name)
#             print(get_folder(target_model_dir), target_model_dir)
#             target_names.append(target.name)
#             target_paths.append(get_image_paths(target.path))
#     return target_names, target_paths


def get_target_paths(data_dir='./data', 
                     model_dir='./model'):
    target_path_dict = dict()

    if not os.path.exists(data_dir):
        data_dir = get_folder(data_dir).path

    data_dir_scanned = list(scandir(data_dir))
    for target in data_dir_scanned:
        if target.name[0] != '.':
            print('   - ' + target.name)
            target_data_dir = '{}/{}'.format(data_dir, target.name)
            get_folder(target_data_dir)
            # target_model_dir = '{}/{}'.format(model_dir, target.name)
            # print(get_folder(target_model_dir), target_data_dir)
            target_path_dict[target.name] = get_image_paths(target_data_dir)
    return target_path_dict


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
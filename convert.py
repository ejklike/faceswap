import argparse
import cv2
import os

from pathlib import Path
from tqdm import tqdm

from plugins.loader import PluginLoader
from plugins.DirectoryProcessor import DirectoryProcessor
from lib.utils import get_target_paths, get_image_paths, get_folder, set_tf_allow_growth

class ConvertProcessor(DirectoryProcessor):
    def prepare_images(self, detector):
        for filename in tqdm(self.read_directory()):
            image = cv2.imread(filename)
            faces = self.get_faces(detector, image)
            yield filename, image, faces

    def convert(self, converter, item):
        try:
            (filename, image, faces) = item

            for idx, face in faces:
                image = converter.patch_image(image, face, size=64)

            output_file = get_folder(self.output_dir) / Path(filename).name
            cv2.imwrite(str(output_file), image)
        except Exception as e:
            print('Failed to convert image: {}. Reason: {}'.format(filename, e))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert a source image to a new one with the face swapped')

    parser.add_argument('-i', '--input-dir',
                        dest="input_dir",
                        default="input",
                        help="Input directory. A directory containing the files \
                        you wish to process. Defaults to 'input'")

    parser.add_argument('-o', '--output-dir',
                        dest="output_dir",
                        default="output",
                        help="Output directory. This is where the converted files will \
                            be stored. Defaults to 'output'")

    parser.add_argument('-m', '--model-dir',
                        dest="model_dir",
                        default="./results/model/",
                        help="Model directory. A directory containing the trained model \
                        you wish to process. Defaults to './results/model/'")

    parser.add_argument('-t', '--trainer',
                        type=str,
                        choices=PluginLoader.get_available_models(), # case sensitive because this is used to load a plug-in.
                        default=PluginLoader.get_default_model(),
                        help="Select the trainer that was used to create the model.")

    parser.add_argument('-tg', '--target',
                        type=str,
                        dest="target",
                        default='A',
                        help="Select target.")

    parser.add_argument('-c', '--converter', ###
                        type=str,
                        choices=("Masked", "Adjust"), # case sensitive because this is used to load a plugin.
                        default="Masked",
                        help="Converter to use.")

    parser.add_argument('-D', '--detector', ###
                        type=str,
                        choices=("hog", "cnn"), # case sensitive because this is used to load a plugin.
                        default="hog",
                        help="Detector to use. 'cnn' detects much more angles but will be much more resource intensive and may fail on large files.")

    parser.add_argument('-b', '--blur-size',
                        type=int,
                        default=2,
                        help="Blur size. (Masked converter only)")

    parser.add_argument('-S', '--seamless',
                        action="store_true",
                        dest="seamless_clone",
                        default=False,
                        help="Use cv2's seamless clone. (Masked converter only)")

    parser.add_argument('-M', '--mask-type',
                        type=str.lower, #lowercase this, because its just a string later on.
                        dest="mask_type",
                        choices=["rect", "facehull", "facehullandrect"],
                        default="facehullandrect",
                        help="Mask to use to replace faces. (Masked converter only)")

    parser.add_argument('-e', '--erosion-kernel-size',
                        dest="erosion_kernel_size",
                        type=int,
                        default=None,
                        help="Erosion kernel size. (Masked converter only). Positive values apply erosion which reduces the edge of the swapped face. Negative values apply dilation which allows the swapped face to cover more space.")

    parser.add_argument('-mh', '--match-histgoram',
                        action="store_true",
                        dest="match_histogram",
                        default=False,
                        help="Use histogram matching. (Masked converter only)")

    parser.add_argument('-sm', '--smooth-mask',
                        action="store_true",
                        dest="smooth_mask",
                        default=True,
                        help="Smooth mask (Adjust converter only)")

    parser.add_argument('-aca', '--avg-color-adjust',
                        action="store_true",
                        dest="avg_color_adjust",
                        default=True,
                        help="Average color adjust. (Adjust converter only)")

    parser.add_argument('-c', '--cuda_visible_devices',
                        type=str,
                        default=None,
                        help="CUDA_VISIBLE_DEVICES value (e.g., -c 0,1)")

    args = parser.parse_args()

    if args.cuda_visible_devices is not None:
        args.gpus = len(args.cuda_visible_devices.split(','))
        if args.allow_growth:
            set_tf_allow_growth(cuda_visible_devices)

    # Original & LowMem models go with Adjust or Masked converter
    # Note: GAN prediction outputs a mask + an image, while other predicts only an image
    # ==> ???
    model_name = args.trainer
    target_image_path_dict = get_target_paths()
    model = PluginLoader.get_model(model_name)(get_folder(args.model_dir), target_image_path_dict, args.gpus)
    if not model.load():
        print('Model Not Found! A valid model must be provided to continue!')
        exit(1)

    conv_name = args.converter
    converter = PluginLoader.get_converter(conv_name)(
        model.converter(args.target),
        trainer=args.trainer,
        blur_size=args.blur_size,
        seamless_clone=args.seamless_clone,
        mask_type=args.mask_type,
        erosion_kernel_size=args.erosion_kernel_size,
        match_histogram=args.match_histogram,
        smooth_mask=args.smooth_mask,
        avg_color_adjust=args.avg_color_adjust
    )

    processor = ConvertProcessor(args.input_dir, args.output_dir)
    for item in processor.prepare_images(args.detector):
        processor.convert(converter, item)
    processor.finalize()
import argparse
import cv2

from pathlib import Path
from tqdm import tqdm
import os
import numpy as np

from plugins.loader import PluginLoader
from plugins.DirectoryProcessor import DirectoryProcessor
from lib.utils import get_folder, get_image_paths
from lib.aligner import get_align_mat


class ExtractProcessor(DirectoryProcessor):
    def extract(self, image, face, size=256):
        def _transform(image, mat, size, padding=0):
            matrix = mat * (size - 2 * padding)
            matrix[:,2] += padding
            return cv2.warpAffine(image, matrix, (size, size))

        alignment = get_align_mat(face)
        extracted = _transform(image, alignment, size, 48)
        return extracted, alignment

    def handle_images(self, detector, debug_landmarks=False):
        for filename in tqdm(self.read_directory()):
            try:
                image = cv2.imread(filename)
                faces = self.get_faces(detector, image)
                process_faces = [(idx, face) for idx, face in faces]

                for idx, face in process_faces:
                    # Draws landmarks for debug
                    if debug_landmarks is True:
                        for (x, y) in face.landmarksXY:
                            cv2.circle(image, (x, y), 2, (0, 0, 255), -1)

                    resized_image, t_mat = self.extract(image, face, size=256)
                    output_file = get_folder(self.output_dir) / Path(filename).stem

                    cv2.imwrite('{}_{}{}'.format(str(output_file), str(idx), Path(filename).suffix), resized_image)
            except Exception as e:
                print('Failed to extract from image: {}. Reason: {}'.format(filename, e))
                pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract the faces from a pictures.')

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

    parser.add_argument('-D', '--detector',
                        type=str,
                        choices=("hog", "cnn"), # case sensitive because this is used to load a plugin.
                        default="hog",
                        help="Detector to use. 'cnn' detects much more angles but will be much more resource intensive and may fail on large files.")

    parser.add_argument('-dl', '--debug-landmarks',
                        action="store_true",
                        dest="debug_landmarks",
                        default=False,
                        help="Draw landmarks for debug.")
    args = parser.parse_args()
    
    extractor = ExtractProcessor(args.input_dir, args.output_dir)
    extractor.handle_images(args.detector, debug_landmarks=args.debug_landmarks)
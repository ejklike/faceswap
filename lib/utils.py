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


def get_image_paths(directory):
    dir_contents = []

    if not os.path.exists(directory):
        directory = get_folder(directory).path

    dir_scanned = list(scandir(directory))
    for x in dir_scanned:
        if any([x.name.lower().endswith(ext) for ext in image_extensions]):
            dir_contents.append(x.path)

    return dir_contents
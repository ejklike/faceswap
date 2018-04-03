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


def load_images( image_paths, convert=None ):
    iter_all_images = ( cv2.imread(fn) for fn in image_paths )
    if convert:
        iter_all_images = ( convert(img) for img in iter_all_images )
    for i,image in enumerate( iter_all_images ):
        if i == 0:
            all_images = numpy.empty( ( len(image_paths), ) + image.shape, dtype=image.dtype )
        all_images[i] = image
    return all_images


# From: https://stackoverflow.com/questions/7323664/python-generator-pre-fetch
import threading
import queue as Queue
class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, prefetch=1): #See below why prefetch count is flawed
        threading.Thread.__init__(self)
        self.queue = Queue.Queue(prefetch)
        self.generator = generator
        self.daemon = True
        self.start()

    def run(self):
        # Put until queue size is reached. Note: put blocks only if put is called while queue has already reached max size
        # => this makes 2 prefetched items! One in the queue, one waiting for insertion!
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def iterator(self):
        while True:
            next_item = self.queue.get()
            if next_item is None:
                break
            yield next_item

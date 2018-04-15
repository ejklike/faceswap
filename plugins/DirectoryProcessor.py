from lib.utils import get_image_paths, get_folder
from lib.detect_faces import detect_faces


class DirectoryProcessor(object):
    '''
    Abstract class that processes a directory of images
    and writes output to the specified folder
    '''

    input_dir = None
    output_dir = None

    images_found = 0
    num_faces_detected = 0
    faces_detected = dict()

    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        print("Input Directory: {}".format(self.input_dir))
        print("Output Directory: {}".format(self.output_dir))

        print('Starting, this may take a while...')
        
        self.output_dir = get_folder(self.output_dir)
        try:
            self.input_dir = get_image_paths(self.input_dir)
        except:
            print('Input directory not found. Please ensure it exists.')
            exit(1)

    def read_directory(self):
        self.images_found = len(self.input_dir)
        return self.input_dir

    def have_face(self, filename):
        return os.path.basename(filename) in self.faces_detected

    def get_faces(self, detector, image):
        faces_count = 0
        faces = detect_faces(image, detector)
        
        for face in faces:
            yield faces_count, face

            self.num_faces_detected += 1
            faces_count += 1

    def finalize(self):
        print('-------------------------')
        print('Images found:        {}'.format(self.images_found))
        print('Faces detected:      {}'.format(self.num_faces_detected))
        print('-------------------------')
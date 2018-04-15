from lib import FaceLandmarksExtractor

def detect_faces(frame, detector):
    for face in FaceLandmarksExtractor.extract(frame, detector):
        x, y, right, bottom, landmarks = face[0][0], face[0][1], face[0][2], face[0][3], face[1]
        yield DetectedFace(image=frame[y: bottom, x: right], 
                           x=x, 
                           w=right - x, 
                           y=y, 
                           h=bottom - y,
                           landmarksXY=landmarks)

class DetectedFace(object):
    def __init__(self, image=None, x=None, w=None, y=None, h=None, landmarksXY=None):
        self.image = image
        self.x = x
        self.w = w
        self.y = y
        self.h = h
        self.landmarksXY = landmarksXY

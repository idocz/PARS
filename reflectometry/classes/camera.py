from utils import *

class Camera(object):
    def __init__(self, width, height, location=None):
        self.width = width
        self.height = height
        self.fovx = np.pi / 4
        # self.fovx = np.pi / 2
        self.fovy = (self.height/self.width) * self.fovx
        if location is None:
            self.location = np.array([0,0,0])
        else:
            self.location = location
    def getPixelDirection(self, i, j):
        x = ((2*i-self.width)/self.width) * np.tan(self.fovx)
        y = -((2 * j - self.height) / self.height) * np.tan(self.fovy)
        return np.array([x, y, -1])

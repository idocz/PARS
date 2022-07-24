import numpy as np

class Ray(object):
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction / np.linalg.norm(direction)
    def copy(self):
        return Ray(np.copy(self.origin), np.copy(self.direction))
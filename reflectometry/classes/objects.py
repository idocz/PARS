from classes.ray import Ray
from utils import *

class Plane(object):
    def __init__(self, d, N, color, emission=0.0, Ks=0.0, n=-1.0):
        self.color = color
        self.emission = emission
        self.Ks = Ks
        self.n = n
        self.N = N
        self.d = d
        self.shape = 0

    def intersect(self, ray:Ray):
        d0 = np.dot(ray.direction,self.N)
        if d0 != 0:
            t = -1*(np.dot(self.N,ray.origin) + self.d)/d0
            if t > eps:
                return t
        return 0
    def normal(self, p0):
        return self.N


class Sphere(object):
    def __init__(self, radius:float, center:np.ndarray, color:np.ndarray, emission=0.0, Ks=0.0, n=-1.0):
        self.color = color
        self.emission = emission
        self.Ks = Ks
        self.n = n
        self.center = center
        self.radius = radius
        self.shape = 1
        if emission == 0:
            self.center[1] = self.radius - 2.5

    def intersect(self, ray:Ray):
        temp = ray.origin - self.center
        b = np.dot(temp*2,ray.direction)
        c_ = np.dot(temp,temp) - self.radius*self.radius
        disc = b*b - 4*c_
        if disc >= 0:
            disc = np.sqrt(disc)
            sol1 = -b + disc
            sol2 = -b - disc
            if sol2 > eps:
                return sol2/2
            elif sol1 > eps:
                return sol1/2
        return 0

    def normal(self, p0):
        res = (p0-self.center)
        res /= np.linalg.norm(res)
        return res

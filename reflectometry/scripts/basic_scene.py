from utils import *
from classes.objects import *
from classes.scene import Scene
from classes.camera import Camera
from time import time
from os.path import join


if __name__ == '__main__':
    pixel_size = 120
    camera = Camera(width=pixel_size, height=pixel_size)
    objects = []
    color_factor = 1
    objects.append(Sphere(1.05, np.array([-0.75, -1.45, -4.4]), np.array([4, 8, 4])*color_factor, 0, 3))
    objects.append(Sphere(0.5, np.array([2.0, -2.05, -3.7]), np.array([10, 10, 1])*color_factor, 0, 1))
    objects.append(Sphere(0.6, np.array([-1.75, -1.95, -3.1]), np.array([4, 4, 12])*color_factor, 0, 1))

    objects.append(Plane(2.5, np.array([0, 1, 0]), np.array([6, 6, 6])*color_factor, 0, 1))
    objects.append(Plane(5.5, np.array([0, 0, 1]), np.array([6, 6, 6])*color_factor, 0, 1))
    objects.append(Plane(2.75, np.array([1, 0, 0]), np.array([10, 2, 2])*color_factor, 0, 1))
    objects.append(Plane(2.75, np.array([-1, 0, 0]), np.array([2, 10, 2])*color_factor, 0, 1))
    objects.append(Plane(3.0, np.array([0, -1, 0]), np.array([6, 6, 6])*color_factor, 0, 1))
    objects.append(Plane(0.5, np.array([0, 0, -1]), np.array([6, 6, 6])*color_factor, 0, 1))
    objects.append(Sphere(0.5, np.array([0, 1.9, -3]), np.array([0, 0, 0]), 10000, 1))

    scene = Scene(objects, camera)
    start = time()
    I_res = scene.render(spp=1024, workers=15)
    print("rendering took:", time()-start)
    # cond = I_res>I_res.max()/2
    # I_res[cond] = 0
    # I_res[cond] = I_res.max() * 2
    # plt.hist(I_res.reshape(-1))
    # plt.show()
    # I_res = (I_res - I_res.min())/ (I_res.max()-I_res.min())
    # I_res = I_res ** 0.4
    # I_res = (I_res * 255).astype(np.uint8)
    I_res[I_res>255] = 255
    I_res = I_res.astype(np.uint8)
    plt.imshow(I_res)
    plt.show()
    plt.imsave(join("outputs/basic_scene.png"), I_res)

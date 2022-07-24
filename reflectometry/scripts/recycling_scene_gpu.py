from utils import *
from classes.objects import *
from classes.scene_recycling import SceneRecycling
from classes.camera import Camera
from time import time
from os.path import join


if __name__ == '__main__':

    Ks0 = 0
    n0 = 10
    Ks = 1
    n = 10000
    pixel_size = 150

    camera = Camera(width=pixel_size, height=pixel_size)
    objects = []
    # objects.append(Sphere(1.05, np.array([-0.75, -1.45, -4.4]), np.array([0.2, 0.4, 0.2]), Ks=Ks, n=n))
    objects.append(Sphere(1.05, np.array([-0.75, -1.45, -4.4]), np.array([0.2, 0.4, 0.2]), Ks=Ks0, n=n0))
    objects.append(Sphere(0.5, np.array([2.0, -2.05, -3.7]), np.array([0.5, 0.5, 0.05]), Ks=0))
    objects.append(Sphere(0.6, np.array([-1.75, -1.95, -3.1]), np.array([0.2, 0.2, 0.6]), Ks=0))

    objects.append(Plane(2.5, np.array([0, 1, 0]), np.array([0.3, 0.3, 0.3])))
    objects.append(Plane(5.5, np.array([0, 0, 1]), np.array([0.3, 0.3, 0.3])))
    objects.append(Plane(2.75, np.array([1, 0, 0]), np.array([0.5, 0.1, 0.1])))
    objects.append(Plane(2.75, np.array([-1, 0, 0]), np.array([0.1, 0.5, 0.1])))
    objects.append(Plane(3.0, np.array([0, -1, 0]), np.array([0.3, 0.3, 0.3])))
    objects.append(Plane(0.5, np.array([0, 0, -1]), np.array([0.3, 0.3, 0.3])))
    objects.append(Sphere(0.5, np.array([0, 1.9, -3]), np.array([0, 0, 0]), 10000))

    scene_rec = SceneRecycling(objects, camera)
    start = time()
    scene_rec.generate_paths(10)
    scene_rec.render()
    print("compiling took",time()-start)
    spp = 2000
    Np = scene_rec.camera.width * scene_rec.camera.height * spp
    start = time()
    scene_rec.generate_paths(spp)
    print("generating took", time()-start)
    print("rendering original scene")
    I_res = scene_rec.render()
    I_res[I_res > 255] = 255
    I_res = I_res.astype(np.uint8)
    plt.figure()
    plt.imshow(I_res)
    plt.title(f"original_scene: Ks0={Ks0}, n0={n0}")
    plt.show()


    print("rendering real scene")
    start = time()
    objects[differ_ind].Ks = Ks
    objects[differ_ind].n = n
    I_res = scene_rec.render()
    print("rendering took:", time() - start)
    I_res[I_res > 255] = 255
    I_res = I_res.astype(np.uint8)
    plt.imshow(I_res)
    plt.title(f"real scene: Ks={Ks}, n={n}")
    plt.show()
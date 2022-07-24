from utils import *
from classes.objects import *
from classes.scene_gpu import SceneGPU
from classes.camera import Camera
from time import time
from os.path import join


if __name__ == '__main__':
    pixel_size = 500
    Ks = 1
    n=-1
    camera = Camera(width=pixel_size, height=pixel_size)
    objects = []
    # objects.append(Sphere(1.05, np.array([-0.75, -1.45, -4.4]), np.array([0.2, 0.4, 0.2]), Ks=Ks, n=n))
    # objects.append(Sphere(0.5, np.array([2.0, -2.05, -3.7]), np.array([0.5, 0.5, 0.05]), Ks=0))
    # objects.append(Sphere(0.6, np.array([-1.75, -1.95, -3.1]), np.array([0.2, 0.2, 0.6]), Ks=0))
    # # objects.append(Sphere(1.4, np.array([-0.0, -1.45, -4.3]), np.array([0.2, 0.4, 0.2]), Ks=Ks, n=n))
    # # objects.append(Sphere(0.5, np.array([2.0, -2.05, -3.1]), np.array([0.5, 0.5, 0.05]), Ks=0))
    # # objects.append(Sphere(0.6, np.array([-1.75, -1.95, -3.1]), np.array([0.2, 0.2, 0.6]), Ks=0))
    #
    # objects.append(Plane(2.5, np.array([0, 1, 0]), np.array([0.3, 0.3, 0.3])))
    # objects.append(Plane(5.5, np.array([0, 0, 1]), np.array([0.3, 0.3, 0.3])))
    # objects.append(Plane(2.75, np.array([1, 0, 0]), np.array([0.5, 0.1, 0.1])))
    # objects.append(Plane(2.75, np.array([-1, 0, 0]), np.array([0.1, 0.5, 0.1])))
    # objects.append(Plane(3.0, np.array([0, -1, 0]), np.array([0.3, 0.3, 0.3])))
    # objects.append(Plane(0.5, np.array([0, 0, -1]), np.array([0.3, 0.3, 0.3])))
    # objects.append(Sphere(0.5, np.array([0, 1.9, -3]), np.array([0, 0, 0]), 10000))
    n_gt = 50
    Ks_gt = 1

    big_sphere_radius = 1.3
    big_sphere_center = np.array([-0.0, big_sphere_radius - 2.5, -4.0])
    objects.append(Sphere(big_sphere_radius, big_sphere_center, np.array([0.2, 0.4, 0.2]), Ks=Ks_gt, n=n_gt))
    N_small_spheres = 14
    R = 2.2
    r = 0.3
    np.random.seed(100)
    colors = np.random.rand(N_small_spheres, 3) * 0.8
    for i in range(N_small_spheres):
        theta = (2 * np.pi) * (i / N_small_spheres)
        center = np.array([R * np.sin(theta), r - 2.5, R * np.cos(theta)])
        center[0] += big_sphere_center[0]
        center[2] += big_sphere_center[2]
        objects.append(Sphere(r, center, colors[i], Ks=0, n=1))

    objects.append(Plane(2.5, np.array([0, 1, 0]), np.array([0.3, 0.3, 0.3])))
    objects.append(Plane(5.5, np.array([0, 0, 1]), np.array([0.3, 0.3, 0.3])))
    objects.append(Plane(2.75, np.array([1, 0, 0]), np.array([0.5, 0.1, 0.1])))
    objects.append(Plane(2.75, np.array([-1, 0, 0]), np.array([0.1, 0.5, 0.1])))
    objects.append(Plane(3.0, np.array([0, -1, 0]), np.array([0.3, 0.3, 0.3])))
    objects.append(Plane(0.5, np.array([0, 0, -1]), np.array([0.3, 0.3, 0.3])))
    objects.append(Sphere(0.5, np.array([0, 1.9, -3]), np.array([0, 0, 0]), 10000))

    Np = 5000
    scene_gpu = SceneGPU(objects, camera)
    start = time()
    I_res = scene_gpu.render(spp=10)
    print("compiling took:", time() - start)
    start = time()

    I_res = scene_gpu.render(spp=Np)
    print("rendering took:", time()-start)
    I_res[I_res>255] = 255
    I_res = I_res.astype(np.uint8)
    plt.imshow(I_res)
    plt.title(f"Basic_GPU: Ks={Ks}, n={n}")
    plt.show()
    plt.imsave(join("outputs/basic_scene.png"), I_res)

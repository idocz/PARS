from utils import *
from classes.objects import *
from classes.scene_recycling import SceneRecycling
from classes.scene_gpu import SceneGPU
from classes.camera import Camera
from time import time
from os.path import join


if __name__ == '__main__':
    Ks0 = 0.3
    n0 = 30
    Ks = 0.9
    n = 40
    render_basic = True
    render_orginal = True
    pixel_size = 200
    camera = Camera(width=pixel_size, height=pixel_size)
    objects = []
    # objects.append(Sphere(1.05, np.array([-0.75, -1.45, -4.4]), np.array([0.2, 0.4, 0.2]), Ks=Ks0, n=n0))
    big_sphere_radius = 1.3
    big_sphere_center = np.array([-0.0, big_sphere_radius-2.5, -4.0])
    objects.append(Sphere(big_sphere_radius, big_sphere_center, np.array([0.2, 0.4, 0.2]), Ks=Ks0, n=n0))
    N_small_spheres = 14
    R = 2.2
    r = 0.3
    colors = np.random.rand(N_small_spheres, 3) * 0.8
    for i in range(N_small_spheres):
        theta = (2*np.pi)*(i/N_small_spheres)
        center = np.array([R*np.sin(theta), r-2.5, R*np.cos(theta)])
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
    #
    # objects.append(Sphere(0.23, np.array([-2.0, -1.95, -2.8]), np.array([0.2, 0.2, 0.6]), Ks=0))
    #
    # objects.append(Sphere(0.23, np.array([-1.5, -1.95, -2.8]), np.array([0.2, 0.6, 0.6]), Ks=0))
    # objects.append(Sphere(0.23, np.array([-1.0, -1.95, -2.8]), np.array([0.6, 0.2, 0.6]), Ks=0))
    # objects.append(Sphere(0.23, np.array([-0.5, -1.95, -2.8]), np.array([0.4, 0.4, 0.6]), Ks=0))
    # objects.append(Sphere(0.23, np.array([0.0, -1.95, -2.8]), np.array([0.1, 0.8, 0.1]), Ks=0))
    # objects.append(Sphere(0.23, np.array([0.5, -1.95, -2.8]), np.array([0.8, 0.1, 0.1]), Ks=0))
    # objects.append(Sphere(0.23, np.array([1, -1.95, -2.8]), np.array([0.8, 0.1, 0.1]), Ks=0))
    # objects.append(Sphere(0.23, np.array([1.5, -1.95, -2.8]), np.array([0.3, 0.4, 0.6]), Ks=0))
    # objects.append(Sphere(0.23, np.array([2.0, -2.05, -2.8]), np.array([0.4, 0.8, 0.4]), Ks=0))



    scene_rec = SceneRecycling(objects, camera)
    scene_basic = SceneGPU(objects, camera)
    start = time()

    print("######### recycling ##########")
    print("COMPILING")
    scene_rec.generate_paths(10)
    scene_rec.render()
    scene_basic.render(spp=10)
    print("compiling took",time()-start)
    spp = 2000

    Np = scene_rec.camera.width * scene_rec.camera.height * spp
    start = time()
    scene_rec.generate_paths(spp)
    print("generating took", time()-start)
    if render_orginal:
        print("rendering original")
        I_rec = scene_rec.render()
        I_rec[I_rec > 255] = 255
        I_rec = I_rec.astype(np.uint8)
        plt.figure()
        plt.imshow(I_rec)
        plt.title(f"recycling original: Ks0={Ks0}, n0={n0}")

    print("rendering real")
    start = time()
    scene_rec.objects[differ_ind].Ks = Ks
    scene_rec.objects[differ_ind].n = n
    I_rec = scene_rec.render()
    I_show = np.copy(I_rec)
    print("rendering took:", time() - start)
    I_show[I_show > 255] = 255
    I_show = I_show.astype(np.uint8)
    plt.figure()
    plt.imshow(I_show)
    plt.title(f"recycling real: Ks0={Ks0}, n0={n0}, Ks={Ks}, n={n}")
    plt.show()
    del(scene_rec)

    if render_basic:
        print("######### basic #########")
        print("COMPILING")
        scene_basic.render(spp=10)
        print("compiling took:", time() - start)
        start = time()
        I_basic1 = scene_basic.render(spp=spp)
        # I_basic2 = scene_basic.render(spp=spp)
        I_show = np.copy(I_basic1)
        I_show[I_show > 255] = 255
        I_show = I_show.astype(np.uint8)
        plt.figure()
        plt.imshow(I_show)
        plt.title(f"basic: Ks={Ks}, n={n}")
        plt.show()

        print("rendering took:", time() - start)
        # print("rel_dist basic vs basic:", rel_dist(I_basic2, I_basic1))
        print("rel_dist rec vs basic:", rel_dist(I_rec, I_basic1))

from utils import *
from classes.objects import *
from classes.scene_ggx import SceneGGX
from classes.camera import Camera
from time import time
from os.path import join


if __name__ == '__main__':
    pixel_size = 500
    Ks = 0.7
    n=30
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
    # objects.append(Sphere(1e5, np.array([1e5 + 1, 40.8, 81.6]),  np.array([.75, .25, .25]), Ks=0)) # Left
    # objects.append(Sphere(1e5, np.array([-1e5 + 99, 40.8, 81.6]), np.array([.25, .25, .75]), Ks=0))# Rght
    # objects.append(Sphere(1e5, np.array([50, 40.8, 1e5]), np.array([.75, .75, .75]), Ks=0)) # Back
    # objects.append(Sphere(1e5, np.array([50, 40.8, -1e5 + 170]), np.array([0.5,0.5,0.5]), Ks=0)) # Frnt
    # objects.append(Sphere(1e5, np.array([50, 1e5, 81.6]), np.array([.75, .75, .75]), Ks=0)) # Botm
    # objects.append(Sphere(1e5, np.array([50, -1e5 + 81.6, 81.6]),  np.array([.75, .75, .75]), Ks=0)) # Top
    # objects.append(Sphere(16.5, np.array([27, 16.5, 47]), np.array([1, 1, 1]) * .999, Ks=1)) # Mirr
    # objects.append(Sphere(600, np.array([50, 681.6 - .27, 81.6]), np.array([0, 0, 0]), 10000)) # Lite
    regular_val = 0.75
    noncolor_val = 0.25
    big_sphere_radius = 1.3
    big_sphere_center = np.array([-0.0, big_sphere_radius - 2.5, -4.0])
    objects.append(Sphere(big_sphere_radius, big_sphere_center, np.array([regular_val, noncolor_val, regular_val]), Ks=Ks, n=n))
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
        objects.append(Sphere(r, center, colors[i], Ks=1, n=-1))


    gray_wall = np.array([regular_val, regular_val, regular_val])
    red_wall = np.array([regular_val, noncolor_val, noncolor_val])
    blue_wall = np.array([noncolor_val, noncolor_val, regular_val])
    objects.append(Plane(2.5, np.array([0, 1, 0]), gray_wall)) # bottom wall
    objects.append(Plane(5.5, np.array([0, 0, 1]), gray_wall))  # forward
    objects.append(Plane(2.75, np.array([1, 0, 0]), red_wall)) #left wall
    objects.append(Plane(2.75, np.array([-1, 0, 0]), blue_wall)) # right wall
    objects.append(Plane(3.0, np.array([0, -1, 0]), gray_wall)) # top wall
    objects.append(Plane(0.5, np.array([0, 0, -1]), gray_wall*0)) # backward
    objects.append(Sphere(0.5, np.array([0, 1.9, -2]), np.array([0, 0, 0]), 5000))
    print(get_interesection_point(objects[-2],objects[-3],objects[-5]))
    exit()
    Np = 5000
    Np_batch = 1000
    batches = Np // Np_batch
    scene_ggx = SceneGGX(objects, camera)
    start = time()
    I_res = scene_ggx.render(spp=10, to_init_rng=True)
    print("compiling took:", time() - start)
    start = time()

    I_res = np.zeros_like(I_res)
    init = True
    for batch in tqdm(range(batches)):
        I_res += scene_ggx.render(spp=Np_batch, to_init_rng=init)
        init = False
    I_res /= batches
    print("rendering took:", time()-start)
    I_res[I_res>255] = 255
    I_res = I_res.astype(np.uint8)
    plt.imshow(I_res)
    plt.title(f"Basic_GPU: Ks={Ks}, n={n}")
    plt.show()
    # plt.imsave(join("outputs/basic_scene.png"), I_res)

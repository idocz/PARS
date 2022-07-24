from utils import *
from classes.objects import *
from classes.scene_ggx import SceneGGX
from classes.camera import Camera
from time import time
from os.path import join
import matplotlib.pyplot as plt
from PIL import Image

if __name__ == '__main__':
    pixel_size = 500
    Ks = 1
    n=-1
    camera = Camera(width=pixel_size, height=pixel_size)
    objects = []
    regular_val = 0.75 * 0.5
    noncolor_val = 0.25 * 0.5
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
    objects.append(Sphere(0.5, np.array([0, 1.9, -2]), np.array([0, 0, 0]), 10000))
    P0 = get_interesection_point(objects[-2],objects[-3],objects[-5])
    Px = get_interesection_point(objects[-2],objects[-3],objects[-4])
    Py = get_interesection_point(objects[-2],objects[-5],objects[-7])
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(*P0, color="blue")
    ax.scatter(*Px, color="green")
    ax.scatter(*Py, color="red")
    plt.show()
    dx = Px - P0
    dist_x = np.linalg.norm(dx)
    dx /= dist_x
    dy = Py - P0
    dist_y = np.linalg.norm(dy)
    dy /= dist_y
    dx = dx.reshape(-1)
    dy = dy.reshape(-1)
    P0 = P0.reshape(-1)
    plane_corners = np.array([[P0[0], P0[1], P0[2], 0],
                              [dx[0], dx[1], dx[2], dist_x],
                              [dy[0], dy[1], dy[2], dist_y]])

    np.random.seed(1000)
    # albedo_image = np.random.rand(3,3,3)
    albedo_image = np.array(Image.open(join("data","eccv.png")))[:,:,:3]
    # albedo_image = np.rot90(albedo_image, k=2)
    albedo_image = (albedo_image - albedo_image.min())/(albedo_image.max()-albedo_image.min())
    albedo_image *= 0.75
    print(plane_corners)
    Np = 2000
    Np_batch = Np
    batches = Np // Np_batch
    scene_ggx = SceneGGX(objects, albedo_image, plane_corners, 20, camera)
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

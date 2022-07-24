from utils import *
from classes.objects import *
from classes.scene_ggx import SceneGGX
from classes.camera import Camera
from time import time
from os.path import join
import matplotlib.pyplot as plt
from PIL import Image

if __name__ == '__main__':
    pixel_size = 100
    camera = Camera(width=pixel_size, height=pixel_size)
    objects = []
    diffusive_scale = 0.5
    regular_val = 0.75 * diffusive_scale
    noncolor_val = 0.25 * diffusive_scale
    big_sphere_radius = 1.8
    big_sphere_center = np.array([-0.0, big_sphere_radius - 2.5, -4.0])
    objects.append(Sphere(big_sphere_radius, big_sphere_center, np.array([regular_val, noncolor_val, regular_val]), Ks=1, n=-1))
    N_small_spheres = 14
    R = 2.5
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
    plane_skew = 0.5
    objects.append(Plane(2.5, np.array([0, 1, 0]), gray_wall)) # bottom wall
    objects.append(Plane(5.5, np.array([0, 0, 1]), gray_wall))  # forward
    objects.append(Plane(2.75+plane_skew, np.array([1, 0, 0]), red_wall)) #left wall
    objects.append(Plane(2.75+plane_skew, np.array([-1, 0, 0]), blue_wall)) # right wall
    objects.append(Plane(3.0, np.array([0, -1, 0]), gray_wall)) # top wall
    objects.append(Plane(0, np.array([0, 0, -1]), gray_wall*0)) # backward

    objects.append(Sphere(0.5, np.array([0, 1.9, -2]), np.array([0, 0, 0]), 1)) # light source
    P0 = get_interesection_point(objects[-2],objects[-3],objects[-5])
    Px = get_interesection_point(objects[-2],objects[-3],objects[-4])
    Py = get_interesection_point(objects[-2],objects[-5],objects[-7])
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(*P0, color="blue")
    # ax.scatter(*Px, color="green")
    # ax.scatter(*Py, color="red")
    # plt.show()
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
    Np = 500
    # Np_batch = 100
    Np_batch = Np
    batches = Np // Np_batch
    scene_ggx = SceneGGX(objects, albedo_image, plane_corners, 20, camera)
    start = time()
    Np_complie = 10
    I_res = scene_ggx.render(spp=Np_complie, to_init_rng=True)
    scene_ggx.generate_paths(spp=Np_complie)
    I_res = scene_ggx.render_recycling()
    init = True
    print("compiling took:", time() - start)

    render_non_recycling = False
    render_recycling = True

    if render_non_recycling:
        I_res_basic = np.zeros_like(I_res)

        start = time()
        for batch in tqdm(range(batches)):
            I_res_basic += scene_ggx.render(spp=Np_batch, to_init_rng=init)
            init = False
        I_res_basic /= batches
        print("rendering took:", time()-start)
        plt.imshow(convert_to_uint8(I_res_basic))
        plt.title(f"Basic_GPU")
        plt.show()

    if render_recycling:
        I_res_recycling = np.zeros_like(I_res)
        grad_recycling = np.zeros_like(albedo_image)
        start = time()
        to_sort = True
        for batch in tqdm(range(batches)):
            scene_ggx.generate_paths(spp=Np_batch, to_sort=to_sort, init_rng=init)
            scene_ggx.update_albedo_image(albedo_image*0)
            I_res_recycling_batch = scene_ggx.render_recycling()
            scene_ggx.update_albedo_image(albedo_image)
            I_res_recycling_batch, grad_recycling_batch = scene_ggx.render_recycling(I_res_recycling_batch)
            I_res_recycling += I_res_recycling_batch
            grad_recycling += grad_recycling_batch
            # I_res_recycling += scene_ggx.render(spp=Np_batch, to_init_rng=init)
            init = False
        I_res_recycling /= batches
        grad_recycling /= batches
        print("rendering recycling took:", time()-start)
        # for max_factor in np.linspace(0.01,0.1,20):
        for max_factor in [0.05]:
            plt.figure()
            plt.imshow(convert_to_uint8(I_res_recycling,max_factor))
            plt.title(f"recycling_GPU: max_factor={max_factor}")
            plt.show()
        # plt.figure()
        # plt.imshow(convert_to_uint8(grad_recycling,1))
        # plt.title(f"recycling_GPU grad: ")
        # plt.show()
        # print(grad_recycling.mean(), grad_recycling.max())
        # plt.figure()
        # plt.hist(grad_recycling[-1])
        # plt.show()
    # plt.imsave(join("outputs/basic_scene.png"), I_res)

    if render_recycling and render_non_recycling:
        scatter_plot_comparison(I_res_basic, I_res_recycling, "basic vs recycling")
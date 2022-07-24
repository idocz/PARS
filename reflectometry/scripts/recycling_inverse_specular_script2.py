from utils import *
from classes.objects import *
# from classes.scene_recycling import SceneRecycling
from classes.scene_recycling_sorting import SceneRecycling
from classes.camera import Camera
from time import time
from classes.tensorboard_wrapper import TensorBoardWrapper
from os.path import join


pixel_size = 60
camera = Camera(width=pixel_size, height=pixel_size)

n_gt = 50
Ks_gt = 0.7

objects = []
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

scene_rec = SceneRecycling(objects, camera)
scene_rec.compile(spp=10, inverse_type="specular")
start = time()
spp_gt = 3000
spp = 1024
scene_rec.generate_paths(spp=spp_gt, init_rng=True)
I_gt = scene_rec.render()
print("rendering took:", time()-start)

step_size_n = 3e-6
max_n = 200
step_size_Ks = 3e-10
iteration = 200

resample_freq = 10
plot_freq = 1
tensorboard = True
if tensorboard:
    tb = TensorBoardWrapper(I_gt)

# initialize scene
res_n = 1
res_Ks = 0.01
objects[differ_ind].n = res_n
objects[differ_ind].Ks = res_Ks

total_grad = np.array([1, 1])
start_loop = time()
for iter in range(iteration):
    start = time()
    if iter % resample_freq == 0:
        print("Generating path...")
        start_gen = time()
        scene_rec.generate_paths(spp, init_rng=False)
        print("generating took:", time() - start_gen)
    dist_n = n_gt - res_n
    dist_Ks = Ks_gt - res_Ks
    print(f"iter {iter}: res_n:{res_n}, res_Ks:{res_Ks}, dist_n:{dist_n}, dist_Ks:{dist_Ks} total_grad:{total_grad[0]:.2e}, {total_grad[1]:.2e}")
    I_res, total_grad = scene_rec.render(I_gt, inverse_type="specular")
    res_n -= step_size_n * total_grad[0]
    res_Ks -= step_size_Ks * total_grad[1]
    if res_n <= 0:
        res_n = 1
    if res_Ks < 0:
        res_Ks = 0.01
    if res_Ks > 1:
        res_Ks = 0.99
    if res_n < 0:
        res_n = 0.01
    if res_n > max_n:
        res_n = max_n
        # update object
    objects[differ_ind].n = res_n
    objects[differ_ind].Ks = res_Ks
    if iter % plot_freq == 0 and tensorboard:
        loss = np.sum((I_res - I_gt)**2)
        rel_dist1 = (np.abs(n_gt-res_n)+np.abs(Ks_gt-res_Ks))/ (n_gt + Ks_gt)
        tb.update(I_res, loss, rel_dist1, res_Ks, res_n, iter)
    print("iteration took:",time()-start)
    print()

rel_dist1 = (np.abs(n_gt-res_n)+np.abs(Ks_gt-res_Ks))/ (n_gt + Ks_gt)
print("###### RECONSTRUCTION HAS FINISHED ######")
print("running time:", time()-start_loop)
print(f"n={res_n} vs n_GT={n_gt}")
print(f"Ks={res_Ks} vs Ks_GT={Ks_gt}")
print("relative distance", rel_dist1)
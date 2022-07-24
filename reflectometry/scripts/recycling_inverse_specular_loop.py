import os, sys
my_lib_path = os.path.abspath('./')
sys.path.append(my_lib_path)
from utils import *
from classes.objects import *
# from classes.scene_recycling import SceneRecycling
from classes.scene_recycling_sorting import SceneRecycling
from classes.camera import Camera
from time import time
from classes.tensorboard_wrapper import TensorBoardWrapper
from os.path import join
from datetime import datetime

cuda.select_device(0)

pixel_size = 60
camera = Camera(width=pixel_size, height=pixel_size)

n_gt = 50
Ks_gt = 0.7

objects = []
diffusive_scale = 0.5
regular_val = 0.75 * diffusive_scale
noncolor_val = 0.25 * diffusive_scale
big_sphere_radius = 1.8
big_sphere_center = np.array([-0.0, big_sphere_radius - 2.5, -4.0])
objects.append(Sphere(big_sphere_radius, big_sphere_center, np.array([0.1, 0.3, 0.1]), Ks=Ks_gt, n=n_gt))
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
    # objects.append(Sphere(r, center, colors[i]))


gray_wall = np.array([regular_val, regular_val, regular_val])
red_wall = np.array([regular_val, noncolor_val, noncolor_val])
blue_wall = np.array([noncolor_val, noncolor_val, regular_val])
plane_skew = 0
objects.append(Plane(2.5, np.array([0, 1, 0]), gray_wall)) # bottom wall
objects.append(Plane(5.5, np.array([0, 0, 1]), gray_wall))  # forward
objects.append(Plane(2.75+plane_skew, np.array([1, 0, 0]), red_wall)) #left wall
objects.append(Plane(2.75+plane_skew, np.array([-1, 0, 0]), blue_wall)) # right wall
objects.append(Plane(3.0, np.array([0, -1, 0]), gray_wall)) # top wall
objects.append(Plane(0, np.array([0, 0, -1]), gray_wall*0)) # backward
objects.append(Sphere(0.5, np.array([0, 1.9, -2]), np.array([0, 0, 0]), 1)) # light source

scene_rec = SceneRecycling(objects, camera)
scene_rec.compile(spp=10, inverse_type="specular")
start = time()
spp_gt = 2048
spp = 2048
N_gt_batches = 10
for i in tqdm(range(N_gt_batches)):
    if i == 0:
        scene_rec.generate_paths(spp=spp_gt, init_rng=True)
        I_gt = scene_rec.render()
    else:
        scene_rec.generate_paths(spp=spp_gt, init_rng=False)
        I_gt += scene_rec.render()
I_gt /= N_gt_batches


print("rendering took:", time()-start)

step_size_n = 8e1
max_n = 200
step_size_Ks = 5e-3

resample_freq = int(sys.argv[1])
runtime = int(sys.argv[2]) #minutes
to_sort = int(sys.argv[3])


plot_freq = 1
tensorboard = True

if tensorboard:
    title =  datetime.now().strftime("%d%m-%H%M-%S")+f"_Nr={resample_freq}_tosort={int(to_sort)}_ss=[{step_size_n:.1e},{step_size_Ks:.1e}]"
    tb = TensorBoardWrapper(I_gt, title= title)

# initialize scene
res_n = 1
res_Ks = 0.01
objects[differ_ind].n = res_n
objects[differ_ind].Ks = res_Ks

total_grad = np.array([1, 1])

# for iter in range(iteration):


start_loop = time()
iteration = 0
while time() - start_loop < runtime*60:
    start = time()
    if iteration % resample_freq == 0:
        print("Generating path...")
        start_gen = time()
        scene_rec.generate_paths(spp, init_rng=False, to_sort=to_sort)
        print("generating took:", time() - start_gen)
    dist_n = n_gt - res_n
    dist_Ks = Ks_gt - res_Ks
    print(f"iter {iteration}: res_n:{res_n}, res_Ks:{res_Ks}, dist_n:{dist_n}, dist_Ks:{dist_Ks} total_grad:{total_grad[0]:.2e}, {total_grad[1]:.2e}")
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
    if iteration % plot_freq == 0 and tensorboard:
        loss = np.sum((I_res - I_gt)**2)
        rel_dist1 = (np.abs(n_gt-res_n)+np.abs(Ks_gt-res_Ks))/ (n_gt + Ks_gt)
        tb.update(I_res, loss, rel_dist1, res_Ks, res_n, iteration)
    print("iteration took:",time()-start)
    print()
    iteration += 1

rel_dist1 = (np.abs(n_gt-res_n)+np.abs(Ks_gt-res_Ks))/ (n_gt + Ks_gt)
print("###### RECONSTRUCTION HAS FINISHED ######")
print("running time:", time()-start_loop)
print(f"n={res_n} vs n_GT={n_gt}")
print(f"Ks={res_Ks} vs Ks_GT={Ks_gt}")
print("relative distance", rel_dist1)
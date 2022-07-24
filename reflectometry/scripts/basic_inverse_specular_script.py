from utils import *
from classes.objects import *
from classes.scene_gpu import SceneGPU
from classes.camera import Camera
from time import time
from classes.tensorboard_wrapper import TensorBoardWrapper
from os.path import join


pixel_size = 150
camera = Camera(width=pixel_size, height=pixel_size)

n_gt = 30
Ks_gt = 0.5

objects = []
objects.append(Sphere(1.05, np.array([-0.75, -1.45, -4.4]), np.array([0.2, 0.4, 0.2]), Ks=Ks_gt, n=n_gt))
objects.append(Sphere(0.5, np.array([2.0, -2.05, -3.7]), np.array([0.5, 0.5, 0.05]), Ks=0))
objects.append(Sphere(0.6, np.array([-1.75, -1.95, -3.1]), np.array([0.2, 0.2, 0.6]), Ks=0))

objects.append(Plane(2.5, np.array([0, 1, 0]), np.array([0.3, 0.3, 0.3])))
objects.append(Plane(5.5, np.array([0, 0, 1]), np.array([0.3, 0.3, 0.3])))
objects.append(Plane(2.75, np.array([1, 0, 0]), np.array([0.5, 0.1, 0.1])))
objects.append(Plane(2.75, np.array([-1, 0, 0]), np.array([0.1, 0.5, 0.1])))
objects.append(Plane(3.0, np.array([0, -1, 0]), np.array([0.3, 0.3, 0.3])))
objects.append(Plane(0.5, np.array([0, 0, -1]), np.array([0.3, 0.3, 0.3])))
objects.append(Sphere(0.5, np.array([0, 1.9, -3]), np.array([0, 0, 0]), 10000))


scene_gpu = SceneGPU(objects, camera)
start = time()
_ = scene_gpu.render(spp=10)
print("compiling took:", time() - start)
start = time()
Np_gt = 3000
Np = 512
I_gt = scene_gpu.render(spp=Np_gt)
print("rendering took:", time()-start)

step_size_n = 3e-6
max_n = 200
step_size_Ks = 3e-10
iteration = 400
plot_freq = 1
tensorboard = True
if tensorboard:
    tb = TensorBoardWrapper(I_gt)

# initialize scene
res_n = 90.0
res_Ks = 0.9
objects[differ_ind].n = res_n
objects[differ_ind].Ks = res_Ks

total_grad = np.array([1, 1])
start_loop = time()
for iter in range(iteration):
    start = time()
    dist_n = n_gt - res_n
    dist_Ks = Ks_gt - res_Ks
    print(f"iter {iter}: res_n:{res_n}, res_Ks:{res_Ks}, dist_n:{dist_n}, dist_Ks:{dist_Ks} total_grad:{total_grad[0]:.2e}, {total_grad[1]:.2e}")
    I_res, total_grad = scene_gpu.render(Np, I_gt, inverse_type="specular", init_rng=False)
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
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
N_exp = 100
# I_gt_list = np.array((N_exp,pixel_size,pixel_size,3),dtype=float)
I_gt_list = []
for n in range(N_exp):
    scene_rec.generate_paths(spp=spp_gt, init_rng=True)
    I_gt_list.append(scene_rec.render()[None])
    print("rendering took:", time()-start)
I_gt_list = np.concatenate(I_gt_list,axis=0)
I_scatter = I_gt_list.reshape(N_exp,-1)
I_mean = np.mean(I_gt_list,axis=0).reshape(-1)
I_std = np.std(I_gt_list,axis=0).reshape(-1)
plt.figure()
# for i in range(I_mean.shape[0]):
#     plt.scatter(i, I_mean[i],color="blue")
#     for n in range(N_exp):
#         plt.scatter(i,I_scatter[n,i], color="red")
#
# plt.show()
# plt.figure()
# N_pix = pixel_size * pixel_size *3
# plt.bar(np.arange(N_pix), I_mean,yerr=I_std)
# plt.show()
cond = I_mean < 1000
plt.scatter(I_mean[cond],I_std[cond])
plt.show()
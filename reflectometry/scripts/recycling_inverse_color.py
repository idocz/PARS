from utils import *
from classes.objects import *
from classes.scene_recycling import SceneRecycling
from classes.camera import Camera
from time import time
from classes.tensorboard_wrapper import TensorBoardWrapper
from os.path import join
print("Reconstrcting wall color using recycling method")

color_gt = np.array([0.1, 0.5, 0.1])

pixel_size = 200
camera = Camera(width=pixel_size, height=pixel_size)
objects = build_scene_objects(Ks0=0, n0=1)
objects[differ_ind].color = color_gt

scene_rec = SceneRecycling(objects, camera)
start = time()
print("COMPILING")
scene_rec.compile(100, inverse_type="color")
print("compiling took:", {time()-start})
start = time()
spp = 1000
scene_rec.generate_paths(spp)
I_gt = scene_rec.render()
print("rendering took:", time()-start)

step_size = 1e-9
resample_freq = 1
iteration = 10
plot_freq = 1
tensorboard = True
if tensorboard:
    tb = TensorBoardWrapper(I_gt)

# initialize scene
res_color = np.array([0.05,0.05,0.05], dtype=np.float64)
scene_rec.objects[differ_ind].color = res_color


init_rng = True
total_grad = np.array([1, 1, 1])
start_loop = time()
for iter in range(iteration):
    start = time()
    dist = color_gt - res_color
    print(f"iter {iter}: res:{res_color},  dist:{dist}, total_grad:{total_grad[0]:.2e}, {total_grad[1]:.2e}, {total_grad[2]:.2e}")
    if iter % resample_freq == 0:
        print("Generating path...")
        start_gen = time()
        scene_rec.generate_paths(spp, init_rng=False)
        print("generating took:",time()-start_gen)
    I_res, total_grad = scene_rec.render(I_gt, inverse_type="color")
    res_color -= step_size * total_grad
    res_color[res_color<0] = 0
    if iter % plot_freq == 0:
        loss = np.sum((I_res - I_gt)**2)
        rel_dist1 = np.sum(np.abs(color_gt-res_color)) / np.sum(color_gt)
        tb.update(I_res, loss, rel_dist1, iter)
    print("iteration took:",time()-start)
    print()

print("###### RECONSTRUCTION HAS FINISHED ######")
print("running time:", time()-start_loop)
print("res_color", res_color)
print("relative distance", rel_dist1)
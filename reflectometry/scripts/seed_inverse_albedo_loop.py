import os, sys
my_lib_path = os.path.abspath('./')
sys.path.append(my_lib_path)
from utils import *
from classes.objects import *
from classes.scene_ggx import SceneGGX
from classes.camera import Camera
from time import time
from classes.tensorboard_wrapper import TensorBoardWrapper
from os.path import join
from PIL import Image
import cv2
from classes.optimizers import MomentumSGD, SGD
from datetime import datetime
print("Reconstrcting wall albedo image using recycling method")

device = 0
cuda.select_device(device)
print("DEVICE:",device)


pixel_size = 400
camera = Camera(width=pixel_size, height=pixel_size, location=np.array([0,0.15,0]))
objects = build_scene_objects_albedo_image()
plane_corners = get_plane_corners(objects)

np.random.seed(1000)
# albedo_image = np.random.rand(3,3,3)
albedo_image = np.array(Image.open(join("data","technion.png")))[:,:,:3]
# albedo_image = np.rot90(albedo_image, k=2)
albedo_image = (albedo_image - albedo_image.min())/(albedo_image.max()-albedo_image.min())
albedo_image *= 0.75
print(albedo_image.shape)
# exit()
im_size = (150,150)
albedo_image = cv2.resize(albedo_image, dsize=im_size, interpolation=cv2.INTER_CUBIC)
# plt.figure()
# plt.imshow(albedo_image)
# plt.show()
print(plane_corners)
spp = 500
scene_ggx = SceneGGX(objects, albedo_image, plane_corners, 20, camera)
start = time()
print("COMPILING")
spp_complie = 10
scene_ggx.generate_paths(spp=spp_complie, init_rng=True)
I_res,_ = scene_ggx.render_recycling(0)
print("compiling took:", {time()-start})
start = time()
init = True
I_gt = np.zeros_like(I_res)
N_batch = 10
for _ in tqdm(range(N_batch)):
    scene_ggx.generate_paths(spp, init_rng=init)
    I_gt += scene_ggx.render_recycling()
I_gt /= N_batch
# plt.figure()
# plt.imshow(convert_to_uint8(I_gt, max_factor=0.05))
# plt.show()
print("rendering took:", time()-start)

step_size = 3e1
resample_freq = int(sys.argv[1])
runtime = int(sys.argv[2]) #minutes
to_sort = int(sys.argv[3])

plot_freq = 1
alpha = 0.9

tensorboard = True
if tensorboard:
    title = datetime.now().strftime("%d%m-%H%M-%S")# + f"_Nr={resample_freq}_tosort={int(to_sort)}_ss={step_size}"

    tb = TensorBoardWrapper(I_gt, title=title)
    tb.writer.add_image(f"albedo_image_gt", np.transpose(albedo_image, (2,0,1)))

# initialize scene
albedo_image_init = np.ones_like(albedo_image)*0.5
scene_ggx.update_albedo_image(albedo_image_init)


init_rng = True
total_grad = 0
albedo_image_res = scene_ggx.albedo_image
# optimizer = MomentumSGD(variables=albedo_image_res, step_size=step_size,alpha=alpha)
optimizer = SGD(variables=albedo_image_res, step_size=step_size)
start_loop = time()
iteration = 0
while time() - start_loop < runtime*60:
    dist = np.linalg.norm(albedo_image_res - albedo_image)
    print(f"iter {iteration}:  dist:{dist}, loss:{np.sum((I_res-I_gt)**2)} grad_norm:{np.linalg.norm(total_grad)}")
    if iteration % resample_freq == 0:
        print("Generating path...")
        start_gen = time()
        scene_ggx.generate_paths(spp, init_rng=False, to_sort=to_sort)
        print("generating took:",time()-start_gen)
    start = time()
    I_res, total_grad = scene_ggx.render_recycling(I_gt)
    print("rendering took:", time() - start)
    optimizer.step(total_grad)


    if iteration % plot_freq == 0:
        loss = np.sum((I_res - I_gt)**2)
        rel_dist1 = np.sum(np.abs(albedo_image_res-albedo_image)) / np.sum(albedo_image)
        tb.update_albedo_image(I_res, loss, rel_dist1, albedo_image_res, iteration)
    iteration += 1
    # print()

print("###### RECONSTRUCTION HAS FINISHED ######")
print("running time:", time()-start_loop)
# print("res_color", res_color)
# print("relative distance", rel_dist1)
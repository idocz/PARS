
import numpy as np
from numba import njit, cuda
from numba.cuda.random import xoroshiro128p_uniform_float64
from classes.objects import Sphere, Plane
import matplotlib.pyplot as plt
from tqdm import tqdm
from tqdm_multiprocess import TqdmMultiProcessPool
import math
import cv2
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.animation as animation

PI_INV = 1/np.pi
eps = 1e-7
rr_stop_prob = 0.05
rr_depth = 5
max_depth = 500
sample_retries = 20
magic_constant = 1
reff_ind = 1.25

direct_illumination = True
max_n_direct = 100

#differntiable
differ_ind = 0
# differ_ind = 18
# differ_ind = 6






# CUDA FUNCTIONS
sample_unifrom = xoroshiro128p_uniform_float64

@cuda.jit(device=True)
def intersect_plane(N, d, origin, direction):
    d0 = dot_3d(direction, N)
    if d0 != 0:
        t = -1 * (dot_3d(N, origin) + d) / d0
        if t > eps:
            return t
    return 0


@cuda.jit(device=True)
def intersect_sphere(center, radius, origin, direction):
    b = (origin[0] - center[0])*direction[0] + (origin[1] - center[1])*direction[1] + (origin[2] - center[2])*direction[2]
    b = 2*b
    c_ = (origin[0] - center[0])**2 + (origin[1] - center[1])**2 + (origin[2] - center[2])**2 - radius * radius
    disc = b * b - 4 * c_
    if disc >= 0:
        disc = math.sqrt(disc)
        sol1 = -b + disc
        sol2 = -b - disc
        if sol2 > eps:
            return sol2 / 2
        elif sol1 > eps:
            return sol1 / 2
    return 0

@cuda.jit(device=True)
def intersect(obj_row, origin, direction):
    if obj_row[0] == 0:
        return intersect_plane(obj_row[1:4], obj_row[4], origin, direction)
    else:
        return intersect_sphere(obj_row[1:4], obj_row[4], origin, direction)

@cuda.jit(device=True)
def intersect_all(object_mat, current_point, direction):
    # Scene Intersection
    mint = 1000.0
    obj_ind = -1
    for ind in range(object_mat.shape[0]):
        t = intersect(object_mat[ind, :5], current_point, direction)
        if t > 0 and t < mint:
            mint = t
            obj_ind = ind

    return obj_ind, mint


@cuda.jit(device=True)
def getPixelDirection(i, j, width, height, res):
    # fovx = np.pi / 4
    fovx = (np.pi / 4)*0.9
    fovy = (height / width) * fovx
    res[0] = ((2*i-width)/width) * math.tan(fovx)
    res[1] = -((2 * j - height) / height) * math.tan(fovy)
    res[2] = -1

@cuda.jit(device=True)
def calc_normal(point, obj_row, res):
    if obj_row[0] == 0:
        assign_3d(res, obj_row[1:4])
    else:
        sub_3d(point, obj_row[1:4], res)
        norm(res)


@cuda.jit(device=True)
def sample_sphere(prop, rng_states, tid, res):
    u1 = sample_unifrom(rng_states, tid)
    u2 = sample_unifrom(rng_states, tid)
    cost = 1 - 2 * u1
    sint = math.sqrt(1- cost*cost)
    phi = 2 * np.pi * u2
    res[0] = prop[3]*sint * math.cos(phi) + prop[0]
    res[1] = prop[3]*sint * math.sin(phi) + prop[1]
    res[2] = prop[3]*cost + prop[2]


@cuda.jit(device=True)
def sample_hemisphere_uniform_cuda(rng_states, tid, res):
    u1 = sample_unifrom(rng_states, tid)
    u2 = sample_unifrom(rng_states, tid)
    r = math.sqrt(1.0 - u1*u1)
    phi = 2 * np.pi * u2
    res[0] = math.cos(phi)*r
    res[1] = math.sin(phi)*r
    res[2] = u1

@cuda.jit(device=True)
def sample_hemisphere_cuda(n, rng_states, tid, res):
    u1 = sample_unifrom(rng_states, tid)
    u2 = sample_unifrom(rng_states, tid)
    cosa = u1 ** (1/(n+1))
    sina = math.sqrt(1 - cosa**2)
    phi = 2 * np.pi * u2
    res[0] = sina * math.cos(phi)
    res[1] = sina * math.sin(phi)
    res[2] = cosa
    return cosa


@cuda.jit(device=True)
def calc_specular_direction(direction, normal, res):
    cost = dot_3d(direction, normal)
    res[0] = direction[0] - normal[0] * (cost * 2)
    res[1] = direction[1] - normal[1] * (cost * 2)
    res[2] = direction[2] - normal[2] * (cost * 2)
    cost = dot_3d(res, normal)
    return cost

@cuda.jit(device=True)
def calc_brdf(cosa, Ks, n):
    brdf_diffusive = (1 / np.pi)
    if Ks == 0:
        return brdf_diffusive
    else:
        if cosa < 0:
            brdf_specular = 0
        else:
            brdf_specular = ((n + 1) / (2 * np.pi)) * cosa ** n

        brdf = Ks * brdf_specular + (1 - Ks) * brdf_diffusive
        return brdf


@cuda.jit(device=True)
def calc_specular_brdf(cosa, n):
    if cosa < 0:
        return 0
    else:
        return ((n + 1) / (2 * np.pi)) * cosa ** n

@cuda.jit(device=True)
def sample_hemisphere_specular_cuda2(n, rng_states, tid, old_direction, res):
    u1 = sample_unifrom(rng_states, tid)
    u2 = sample_unifrom(rng_states, tid)
    cos_theta = u1 ** (1 / (n + 1))
    phi = 2 * np.pi * u2
    change_direction(old_direction, cos_theta, phi, res)
    return cos_theta


@cuda.jit(device=True)
def change_coordinate_system(sampleDir, up_vector, rotX, rotY, res):
    res[0] = rotX[0] * sampleDir[0] + rotY[0] * sampleDir[1] + up_vector[0] * sampleDir[2]
    res[1] = rotX[1] * sampleDir[0] + rotY[1] * sampleDir[1] + up_vector[1] * sampleDir[2]
    res[2] = rotX[2] * sampleDir[0] + rotY[2] * sampleDir[1] + up_vector[2] * sampleDir[2]


@cuda.jit(device=True)
def change_direction(old_direction, cos_theta, phi, res):
    sin_theta = math.sqrt(1 - cos_theta * cos_theta)
    sin_phi = math.sin(phi)
    cos_phi = math.cos(phi)
    if abs(old_direction[2]) > 0.99999:  # |z| ~ 1
        z_sign = sign(old_direction[2])
        res[0] = sin_theta * cos_phi
        res[1] = z_sign * sin_theta * sin_phi
        res[2] = z_sign * cos_theta
    else:
        denom = math.sqrt(1 - old_direction[2] ** 2)
        z_cos_phi = old_direction[2] * cos_phi
        res[0] = (sin_theta * (old_direction[0] * z_cos_phi - old_direction[1] * sin_phi) / denom) + \
                 old_direction[0] * cos_theta
        res[1] = (sin_theta * (old_direction[1] * z_cos_phi + old_direction[0] * sin_phi) / denom) + \
                 old_direction[1] * cos_theta
        res[2] = old_direction[2] * cos_theta - denom * sin_theta * cos_phi


@cuda.jit(device=True)
def ons_cuda(v1, v2, v3):
    if abs(v1[0]) > abs(v1[1]):
        invLen = 1/ math.sqrt(v1[0]**2 + v1[2]**2)
        v2[0] = -v1[2]*invLen
        v2[1] = 0
        v2[2] = v1[0]*invLen
    else:
        invLen = 1/math.sqrt(v1[1]**2 + v1[2]**2)
        v2[0] = 0
        v2[1] = v1[2]*invLen
        v2[2] = -v1[1]*invLen
    v3[0] = v1[1]*v2[2] - v1[2]*v2[1]
    v3[1] = v1[2]*v2[0] - v1[0]*v2[2]
    v3[2] = v1[0]*v2[1] - v1[1]*v2[0]


@cuda.jit(device=True)
def dot_3d(a,b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]


@cuda.jit(device=True)
def assign_value_3d(a,val):
    a[0] = val
    a[1] = val
    a[2] = val


@cuda.jit(device=True)
def assign_3d(a,b):
    a[0] = b[0]
    a[1] = b[1]
    a[2] = b[2]


@cuda.jit(device=True)
def multiply_by_scalar(a,val):
    a[0] *= val
    a[1] *= val
    a[2] *= val

@cuda.jit(device=True)
def sign(a):
    if a >= 0:
        return 1
    else:
        return -1


@cuda.jit(device=True)
def sub_3d(a,b, res):
    res[0] = a[0] - b[0]
    res[1] = a[1] - b[1]
    res[2] = a[2] - b[2]


@cuda.jit(device=True)
def calc_direction(source, dest, direction):
    direction[0] = dest[0] - source[0]
    direction[1] = dest[1] - source[1]
    direction[2] = dest[2] - source[2]
    distance = math.sqrt(direction[0]**2 + direction[1]**2 + direction[2]**2)
    direction[0] = direction[0] / distance
    direction[1] = direction[1] / distance
    direction[2] = direction[2] / distance
    return distance

@cuda.jit(device=True)
def calc_distance(source, dest):
    return math.sqrt((dest[0] - source[0])**2 + ( dest[1] - source[1])**2 + (dest[2] - source[2])**2)



@cuda.jit(device=True)
def step_in_direction(current_point, direction, step_size):
    current_point[0] += step_size * direction[0]
    current_point[1] += step_size * direction[1]
    current_point[2] += step_size * direction[2]



@cuda.jit(device=True)
def norm(vec):
    length = math.sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2])
    vec[0] = vec[0] / length
    vec[1] = vec[1] / length
    vec[2] = vec[2] / length

@cuda.jit(device=True)
def print_3d(a):
    return print(a[0],a[1],a[2])

@cuda.jit(device=True)
def get_pixel_albedo(current_point, albedo_image, plance_corners, obj_color):

    j = ((current_point[0]-plance_corners[0,0])*plance_corners[1,0] +
          (current_point[1]-plance_corners[0,1])*plance_corners[1,1] +
          (current_point[2]-plance_corners[0,2])*plance_corners[1,2])/plance_corners[1,3]

    i = ((current_point[0] - plance_corners[0, 0]) * plance_corners[2, 0] +
             (current_point[1] - plance_corners[0, 1]) * plance_corners[2, 1] +
             (current_point[2] - plance_corners[0, 2]) * plance_corners[2, 2]) / plance_corners[2, 3]
    # if  j >=1 or j < 0 :
    #     print(i,j)
    i = int(i * albedo_image.shape[0])
    j = int(j * albedo_image.shape[1])
    # if i >= albedo_image.shape[0]:
    #     i = albedo_image.shape[0] - 1
    # if j >= albedo_image.shape[1]:
    #     j = albedo_image.shape[1] - 1

    # if i < 0:
    #     i = 0
    # if j < 0:
    #     j = 0

    if i < 0 or j < 0 or i >= albedo_image.shape[0] or j >= albedo_image.shape[1]:
        obj_color[0] = 0
        obj_color[1] = 0
        obj_color[2] = 0
        # print_3d(current_point)
        return -1,-1
    else:
        assign_3d(obj_color, albedo_image[i,j])
        return i, j

@cuda.jit(device=True)
def transform_1D_to_3D(ind, Xshape, Yshape, Zshape):
    k = ind // (Xshape * Yshape)
    temp = ind - (k * Xshape * Yshape)
    j = temp // Xshape
    i = temp % Xshape
    return i,j,k
####### CPU FUNCTION ########

def build_scene_objects(Ks0, n0):
    objects = []
    # objects.append(Sphere(1.05, np.array([-0.75, -1.45, -4.4]), np.array([0.2, 0.4, 0.2]), Ks=Ks0, n=n0))
    big_sphere_radius = 1.3
    big_sphere_center = np.array([-0.0, big_sphere_radius-2.5, -4.0])
    objects.append(Sphere(big_sphere_radius, big_sphere_center, np.array([0.2, 0.4, 0.2]), Ks=Ks0, n=n0))
    N_small_spheres = 14
    R = 2.2
    r = 0.3
    colors = np.random.rand(N_small_spheres, 3) * 0.5
    for i in range(N_small_spheres):
        theta = (2*np.pi)*(i/N_small_spheres)
        center = np.array([R*np.sin(theta), r-2.5, R*np.cos(theta)])
        center[0] += big_sphere_center[0]
        center[2] += big_sphere_center[2]
        objects.append(Sphere(r, center, np.array([0.1,0.1,0.8]), Ks=0, n=1))
    objects.append(Plane(2.5, np.array([0, 1, 0]), np.array([0.3, 0.3, 0.3])))
    objects.append(Plane(5.5, np.array([0, 0, 1]), np.array([0.3, 0.3, 0.3])))
    objects.append(Plane(2.75, np.array([1, 0, 0]), np.array([0.5, 0.1, 0.1])))
    objects.append(Plane(2.75, np.array([-1, 0, 0]), np.array([0.1, 0.5, 0.1])))
    objects.append(Plane(3.0, np.array([0, -1, 0]), np.array([0.3, 0.3, 0.3])))
    objects.append(Plane(0.5, np.array([0, 0, -1]), np.array([0.3, 0.3, 0.3])))
    objects.append(Sphere(0.5, np.array([0, 1.9, -3]), np.array([0, 0, 0]), 10000))
    return objects

def build_scene_objects_albedo_image():
    objects = []
    diffusive_scale = 0.5
    regular_val = 0.75 * diffusive_scale
    noncolor_val = 0.25 * diffusive_scale
    big_sphere_radius = 1.8
    big_sphere_center = np.array([-0.0, big_sphere_radius - 2.5, -4.0])
    objects.append(
        Sphere(big_sphere_radius, big_sphere_center, np.array([regular_val, noncolor_val, regular_val]), Ks=1, n=-1))
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
    objects.append(Plane(2.5, np.array([0, 1, 0]), gray_wall))  # bottom wall
    objects.append(Plane(5.5, np.array([0, 0, 1]), gray_wall))  # forward
    objects.append(Plane(2.75 + plane_skew, np.array([1, 0, 0]), red_wall))  # left wall
    objects.append(Plane(2.75 + plane_skew, np.array([-1, 0, 0]), blue_wall))  # right wall
    objects.append(Plane(3.0, np.array([0, -1, 0]), gray_wall))  # top wall
    objects.append(Plane(0.0, np.array([0, 0, -1]), gray_wall * 0))  # backward

    objects.append(Sphere(0.5, np.array([0, 1.9, -2]), np.array([0, 0, 0]), 1))  # light source
    return objects

def build_scene_objects_old():
    objects = []
    color_gt = np.array([0.1, 0.5, 0.1])
    objects.append(Sphere(1.05, np.array([-0.75, -1.45, -4.4]), np.array([0.2, 0.4, 0.2]), Ks=1, n=-1))
    objects.append(Sphere(0.5, np.array([2.0, -2.05, -3.7]), np.array([0.5, 0.5, 0.05]), Ks=0))
    objects.append(Sphere(0.6, np.array([-1.75, -1.95, -3.1]), np.array([0.2, 0.2, 0.6]), Ks=0))
    objects.append(Plane(2.5, np.array([0, 1, 0]), np.array([0.3, 0.3, 0.3])))
    objects.append(Plane(5.5, np.array([0, 0, 1]), np.array([0.3, 0.3, 0.3])))
    objects.append(Plane(2.75, np.array([1, 0, 0]), np.array([0.5, 0.1, 0.1])))
    objects.append(Plane(2.75, np.array([-1, 0, 0]), color_gt))
    objects.append(Plane(3.0, np.array([0, -1, 0]), np.array([0.3, 0.3, 0.3])))
    objects.append(Plane(0.5, np.array([0, 0, -1]), np.array([0.3, 0.3, 0.3])))
    objects.append(Sphere(0.5, np.array([0, 1.9, -3]), np.array([0, 0, 0]), 10000))
    return objects

def rel_dist(A,B):
    A = A.reshape(-1)
    B = B.reshape(-1)
    return (np.sum(np.abs(A-B))/np.sum(np.abs(A)))

def rel_bias(A,B):
    A = A.reshape(-1)
    B = B.reshape(-1)
    return (np.sum(np.abs(A))-np.sum(np.abs(B)))/np.sum(np.abs(A))

@njit()
def ons(v1):
    if np.abs(v1[0]) > np.abs(v1[1]):
        invLen = 1/np.sqrt(v1[0]**2 + v1[2]**2)
        v2 = np.array([-v1[2]*invLen, 0.0, v1[0]*invLen])
    else:
        invLen = 1/np.sqrt(v1[1]**2 + v1[2]**2)
        v2 = np.array([0.0, v1[2]*invLen, -v1[1]*invLen])
    v3 = np.empty_like(v2)
    v3[0] = v1[1]*v2[2] - v1[2]*v2[1]
    v3[1] = v1[2]*v2[0] - v1[0]*v2[2]
    v3[2] = v1[0]*v2[1] - v1[1]*v2[0]
    return v2, v3




def get_images_from_TB(exp_dir, label):
    ea = event_accumulator.EventAccumulator(exp_dir, size_guidance={"images": 0})
    ea.Reload()
    img_list = []
    for img_bytes in ea.images.Items(label):
        img = np.frombuffer(img_bytes.encoded_image_string, dtype=np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.array(img)
        img_list.append(img)
    return img_list

def get_scalars_from_TB(exp_dir, label):
    ea = event_accumulator.EventAccumulator(exp_dir, size_guidance={"scalars": 0})
    ea.Reload()
    return ea.scalars.Items(label)

def animate(image_list, interval, repeat_delay=250, output_name=None):
    ims = []
    fig = plt.figure()
    for image in image_list:
        plt.axis("off")
        im = plt.imshow(image, animated=True, cmap="gray")

        ims.append([im])


    ani = animation.ArtistAnimation(fig, ims, interval=interval, blit=True,
                                    repeat_delay=repeat_delay)
    if not output_name is None:
        Writer = animation.writers['ffmpeg']
        writer = animation.FFMpegWriter(fps=30)
        # writer = animation.PillowWriter(fps=15)
        ani.save(output_name, writer=writer)
    plt.title("Reconstructed Vs Ground Truth")
    plt.show()

def get_interesection_point(plane1,plane2,plane3):
    planes = [plane1, plane2, plane3]
    b = np.array([-plane.d for plane in planes]).reshape(-1,1)
    normals = [plane.N.reshape(1,-1) for plane in planes]
    A = np.concatenate(normals, axis=0)

    return np.linalg.inv(A)@ b

def scatter_plot_comparison(signal1, signal2, title):
    plt.figure()
    X = signal1.reshape(-1)
    Y = signal2.reshape(-1)
    max_val = np.max([X.max(), Y.max()])
    min_val = np.min([X.min(), Y.min()])
    mask = X != 0
    rel_err = np.sum(np.abs(X[mask] - Y[mask])) / np.sum(np.abs(X[mask]))
    print(f"{title} err: {rel_err}")
    plt.scatter(X, Y)
    plt.plot([min_val, max_val], [min_val,max_val])
    plt.title(title)

def convert_to_uint8(img, max_factor=0.1, gamma=None):
    img_new = np.copy(img)
    cond = img_new >= max_factor*np.max(img_new)
    # cond = img_new >= 0.0255
    img_new[cond] = 0
    img_new = (img_new - img_new.min())/(img_new.max() - img_new.min())
    # img_new = img_new ** (0.7)
    # img_new *= 10000
    # print(img_new.max())
    if gamma is not None:
        img_new = img_new**gamma
    img_new *= 255
    img_new[cond] = 255
    # img_new[img_new > 255] = 255
    img_new = img_new.astype(np.uint8)
    return img_new

def get_plane_corners(objects):
    P0 = get_interesection_point(objects[-2], objects[-3], objects[-5])
    Px = get_interesection_point(objects[-2], objects[-3], objects[-4])
    Py = get_interesection_point(objects[-2], objects[-5], objects[-7])
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
    return plane_corners
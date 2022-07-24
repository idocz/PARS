from utils import *
from classes.ray import Ray
import multiprocessing as mp
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states
from time import time
class SceneRecycling(object):
    def __init__(self, objects, camera):
        self.objects = objects
        self.camera = camera

        self.dobject_mat = None
        self.dcamera_location = cuda.to_device(camera.location)
        self.rng_states = None
        self.ddepth_inds = None
        self.dhit_points = None
        self.dhit_objects = None
        self.job_inds = None
        self.Np = 0
        @cuda.jit()
        def calc_total_reflection(job_inds, object_mat, camera_location, width, height, spp, rng_states, depth_sizes):
            tid = cuda.grid(1)
            if tid < job_inds.shape[0]:
                j,i = job_inds[tid]
                current_point = cuda.local.array(3, dtype=np.float64)
                direction = cuda.local.array(3, dtype=np.float64)
                specular_direction = cuda.local.array(3, dtype=np.float64)
                normal = cuda.local.array(3, dtype=np.float64)
                sampleDir = cuda.local.array(3, dtype=np.float64)
                rotX = cuda.local.array(3, dtype=np.float64)
                rotY = cuda.local.array(3, dtype=np.float64)
                assign_3d(current_point, camera_location)
                getPixelDirection(i, j, width, height, direction)
                calc_direction(camera_location, direction, direction)
                depth = 0
                while True:
                    # Russian Roulette
                    if depth >= rr_depth:
                        if sample_unifrom(rng_states, tid) <= rr_stop_prob:
                            break

                    # Scene Intersection
                    obj_ind, mint = intersect_all(object_mat, current_point, direction)

                    # if ray didn't intersect anything
                    if obj_ind == -1:
                        print("bug")
                        return

                    # stepping towards the intersected object
                    step_in_direction(current_point, direction, mint)


                    # getting object properties
                    calc_normal(current_point, object_mat[obj_ind], normal)
                    emission = object_mat[obj_ind, 8]


                    # if we have reached the light source, add the contribution and terminate
                    if emission > 0:
                        break

                    obj_Ks = object_mat[obj_ind, 9]  # specular constant
                    obj_n = object_mat[obj_ind, 10]  # glossiness constant

                    # determining the intersection brdf type
                    diffusive = True
                    calc_specular_direction(direction, normal, specular_direction)
                    if obj_Ks > 0:
                        if obj_Ks == 1 or sample_unifrom(rng_states, tid) <= obj_Ks:
                            diffusive = False
                            if obj_n != -1:  # Specular but not perfect
                                sample_infinte_loop = True
                                for _ in range(sample_retries):
                                    ons_cuda(specular_direction, rotX, rotY)
                                    sample_hemisphere_cuda(obj_n, rng_states, tid, sampleDir)
                                    change_coordinate_system(sampleDir, specular_direction, rotX, rotY, direction)
                                    cost = dot_3d(normal, direction)
                                    if cost >= 0:
                                        sample_infinte_loop = False
                                        break
                                if sample_infinte_loop:
                                    break


                            else:  # Perfect specular
                                assign_3d(direction, specular_direction)

                    if diffusive:  # Diffusive reflection
                        # indirect illumination
                        ons_cuda(normal, rotX, rotY)
                        sample_hemisphere_cuda(1, rng_states, tid, sampleDir) # importance sampling the cosine term
                        change_coordinate_system(sampleDir, normal, rotX, rotY, direction)  # rotating hemisphere vector


                    # increasing the ray depth
                    depth += 1


                    # direct illumination (just sampling for rng_states sync)
                    direct_flag = diffusive or (obj_n != -1 and obj_n < max_n_direct)
                    if direct_illumination and direct_flag:
                        sample_sphere(object_mat[-1, 1:5], rng_states, tid, rotX)  # using temp rotX for light location

                # track depth for each path
                depth_sizes[tid] = depth

        @cuda.jit()
        def generate_paths_cuda(job_inds, object_mat, camera_location, width, height, spp, rng_states, depth_inds, hit_points,
                                hit_objects):
            tid = cuda.grid(1)
            if tid < job_inds.shape[0]:
                j,i = job_inds[tid]
                current_point = cuda.local.array(3, dtype=np.float64)
                direction = cuda.local.array(3, dtype=np.float64)
                specular_direction = cuda.local.array(3, dtype=np.float64)
                normal = cuda.local.array(3, dtype=np.float64)
                sampleDir = cuda.local.array(3, dtype=np.float64)
                rotX = cuda.local.array(3, dtype=np.float64)
                rotY = cuda.local.array(3, dtype=np.float64)
                assign_3d(current_point, camera_location)
                getPixelDirection(i, j, width, height, direction)
                calc_direction(camera_location, direction, direction)
                depth_start_ind = depth_inds[tid]
                total_depth = depth_inds[tid+1] - depth_start_ind
                depth = 0
                while True:
                    depth_ind = depth_start_ind + depth

                    # Russian Roulette
                    if depth >= rr_depth:
                        if sample_unifrom(rng_states, tid) <= rr_stop_prob:
                            if total_depth != depth:
                                print("bug total_depth != depth")
                            break

                    # Scene Intersection
                    obj_ind, mint = intersect_all(object_mat, current_point, direction)

                    # if ray didn't intersect anything (does not suppose to happened)

                    # stepping towards the intersected object
                    step_in_direction(current_point, direction, mint)

                    # # saving path point and object
                    # assign_3d(hit_points[0, :, depth_ind], current_point)
                    # hit_objects[0, depth_ind] = obj_ind

                    # getting object properties
                    calc_normal(current_point, object_mat[obj_ind], normal)
                    emission = object_mat[obj_ind, 8]
                    # if we have reached the light source, add the contribution and terminate
                    if emission > 0:
                        break

                    obj_Ks = object_mat[obj_ind, 9]  # specular constant
                    obj_n = object_mat[obj_ind, 10]  # glossiness constant

                    # determining the intersection brdf type
                    diffusive = True
                    calc_specular_direction(direction, normal, specular_direction)
                    if obj_Ks > 0:
                        if obj_Ks == 1 or sample_unifrom(rng_states, tid) <= obj_Ks:
                            diffusive = False
                            hit_objects[1, depth_ind] = 0
                            if obj_n != -1:  # Specular but not perfect
                                count = 0
                                sample_infinte_loop = True
                                for _ in range(sample_retries):
                                    ons_cuda(specular_direction, rotX, rotY)
                                    sample_hemisphere_cuda(obj_n, rng_states, tid, sampleDir)
                                    change_coordinate_system(sampleDir, specular_direction, rotX, rotY, direction)
                                    cost = dot_3d(normal, direction)
                                    if cost >= 0:
                                        sample_infinte_loop = False
                                        break
                                if sample_infinte_loop:
                                    print("sample_infinte_loop")
                                    break


                            else:  # Perfect specular
                                assign_3d(direction, specular_direction)

                    if diffusive:  # Diffusive reflection
                        # indirect illumination
                        ons_cuda(normal, rotX, rotY)
                        sample_hemisphere_cuda(1, rng_states, tid, sampleDir)  # importance sampling the cosine term
                        change_coordinate_system(sampleDir, normal, rotX, rotY, direction)  # rotating hemisphere vector
                        hit_objects[1,depth_ind] = 1

                    # # saving path point and object
                    assign_3d(hit_points[0, :, depth_ind], current_point)
                    hit_objects[0, depth_ind] = obj_ind


                    # direct illumination
                    direct_flag = diffusive or (obj_n != -1 and obj_n < max_n_direct)
                    if direct_illumination and direct_flag:
                        sample_sphere(object_mat[-1, 1:5], rng_states, tid, rotX)  # using temp rotX for light location
                        assign_3d(hit_points[1, :, depth_ind], rotX) # save light location
                        distance = calc_direction(current_point, rotX, rotY ) # using rotY for light direction
                        direct_ind, mint = intersect_all(object_mat, current_point, rotY)
                        # direct_ind, mint = 9, 1
                        cosl = 0
                        cost = 0
                        if direct_ind == object_mat.shape[0] - 1:  # light source visibility test
                            rotX[0] = current_point[0] + mint * rotY[0]
                            rotX[1] = current_point[1] + mint * rotY[1]
                            rotX[2] = current_point[2] + mint * rotY[2]
                            cost = dot_3d(rotY, normal)  # surface cosine
                            calc_normal(rotX, object_mat[direct_ind], normal)
                            cosl = -dot_3d(rotY, normal)  # light source cosine

                        if cosl <= 0 or cost <= 0:
                            assign_value_3d(hit_points[1, :, depth_ind], 0) # remove light location
                    # increasing the ray depth
                    depth += 1
                # print(depth-total_depth, )

        @cuda.jit()
        def render_cuda(job_inds, Ks0, n0, hit_points, hit_objects, depth_inds, object_mat, camera_location, width, height, spp
                        , I_res):
            tid = cuda.grid(1)
            if tid < job_inds.shape[0]:
                j,i = job_inds[tid]
                # Local arrays
                res_color = cuda.local.array(3, dtype=np.float64)
                total_color = cuda.local.array(3, dtype=np.float64)
                current_point = cuda.local.array(3, dtype=np.float64)
                temp_point = cuda.local.array(3, dtype=np.float64)
                direction = cuda.local.array(3, dtype=np.float64)
                specular_direction = cuda.local.array(3, dtype=np.float64)
                normal = cuda.local.array(3, dtype=np.float64)
                assign_value_3d(res_color, 1)
                assign_value_3d(total_color, 0)
                assign_3d(current_point, camera_location)
                getPixelDirection(i, j, width, height, direction)
                calc_direction(camera_location, direction, direction)
                direct_flag = False
                depth_start_ind = depth_inds[tid]
                total_depth = depth_inds[tid+1] - depth_start_ind
                rrFactor = 1.0
                for depth in range(total_depth):
                    depth_ind = depth_start_ind + depth
                    # Russian Roulette
                    if depth >= rr_depth:
                        rrFactor = 1.0 / (1.0 - rr_stop_prob)

                    # Scene Intersection
                    mint = calc_direction(current_point, hit_points[0, :, depth_ind], direction)
                    assign_3d(current_point, hit_points[0, :, depth_ind])
                    obj_ind = hit_objects[0, depth_ind]
                    diffusive = hit_objects[1, depth_ind]
                    # if ray didn't intersect anything
                    if obj_ind == -1:
                        print("bug")
                        return

                    # getting object properties
                    calc_normal(current_point, object_mat[obj_ind], normal)
                    emission = object_mat[obj_ind, 8]
                    obj_color = object_mat[obj_ind, 5:8]
                    # if we have reached the light source, add the contribution and terminate
                    if emission > 0:
                        if not direct_illumination or not direct_flag:
                            multiply_by_scalar(res_color, emission * rrFactor)
                        break

                    obj_Ks = object_mat[obj_ind, 9]  # specular constant
                    obj_n = object_mat[obj_ind, 10]  # glossiness constant
                    calc_specular_direction(direction, normal, specular_direction) # calculate specular direction

                    multiply_by_scalar(res_color, rrFactor)
                    if diffusive:  # Diffusive reflection
                        # adding the contribution of the object color (cosine term was canceled by MC)
                        if obj_ind == differ_ind:
                            multiply_by_scalar(res_color, ((1-obj_Ks)/(1-Ks0)))
                        res_color[0] *= obj_color[0]
                        res_color[1] *= obj_color[1]
                        res_color[2] *= obj_color[2]
                    elif obj_ind==differ_ind:
                        multiply_by_scalar(res_color, (obj_Ks / Ks0))

                    assign_3d(temp_point, hit_points[1,:,depth_ind])
                    direct_flag = diffusive or (obj_n != -1 and obj_n < max_n_direct)
                    if direct_illumination and direct_flag and (temp_point[0] != 0 or temp_point[1] != 0 or temp_point[2] != 0):
                        distance = calc_direction(current_point, temp_point, direction)
                        mint = intersect_sphere(object_mat[-1, 1:4], object_mat[-1, 4], current_point, direction)
                        temp_point[0] = current_point[0] + mint * direction[0]
                        temp_point[1] = current_point[1] + mint * direction[1]
                        temp_point[2] = current_point[2] + mint * direction[2]
                        cost = dot_3d(direction, normal)  # surface cosine
                        calc_normal(temp_point, object_mat[-1], normal)  # using normal as temp for light normal
                        cosl = -dot_3d(direction, normal)  # light source cosine
                        # emission * r^-2 * cos(diffuse_angle) * cos(light_angle) * brdf * 1/pdf
                        temp = object_mat[-1, 8] * (1 / (distance * distance)) * cosl
                        rad_squared = object_mat[-1, 4] * object_mat[-1, 4]
                        if diffusive:
                            temp *= 2 * rad_squared * cost
                        else:
                            cosa = dot_3d(direction, specular_direction)
                            brdf = calc_specular_brdf(cosa, obj_n)
                            temp *= brdf * 2 * np.pi * rad_squared

                        total_color[0] += res_color[0] * temp
                        total_color[1] += res_color[1] * temp
                        total_color[2] += res_color[2] * temp

                    # # Recycling Correction Factor
                    if obj_ind == differ_ind and depth < total_depth - 1 and not diffusive:
                        calc_direction(current_point, hit_points[0, :, depth_ind + 1], direction)  # using rotY as temp
                        cosa = dot_3d(direction, specular_direction)
                        brdf_specular = calc_specular_brdf(cosa, obj_n)
                        brdf_specular0 = calc_specular_brdf(cosa, n0)
                        if obj_n == -1:
                            multiply_by_scalar(res_color, cosa>0.99)
                        else:
                            multiply_by_scalar(res_color, brdf_specular/brdf_specular0)

                # atomically adding the final contribution of the ray to the final image
                if direct_illumination :
                    total_color[0] += res_color[0]
                    total_color[1] += res_color[1]
                    total_color[2] += res_color[2]
                    cuda.atomic.add(I_res, (j, i, 0), total_color[0])
                    cuda.atomic.add(I_res, (j, i, 1), total_color[1])
                    cuda.atomic.add(I_res, (j, i, 2), total_color[2])
                else:
                    cuda.atomic.add(I_res, (j, i, 0), res_color[0])
                    cuda.atomic.add(I_res, (j, i, 1), res_color[1])
                    cuda.atomic.add(I_res, (j, i, 2), res_color[2])
        @cuda.jit()
        def render_inverse_color_cuda(Ks0, n0, hit_points, hit_objects, depth_inds, object_mat, camera_location, width,
                                      height, spp, rng_states, I_diff, total_grad):
            i, j, s = cuda.grid(3)
            tid = height * width * s + width * j + i
            if i < width and j < height and s < spp:
                # Local arrays
                res_color = cuda.local.array(3, dtype=np.float64)
                grad = cuda.local.array(3, dtype=np.float64)
                current_point = cuda.local.array(3, dtype=np.float64)
                direction = cuda.local.array(3, dtype=np.float64)
                temp_point = cuda.local.array(3, dtype=np.float64)
                specular_direction = cuda.local.array(3, dtype=np.float64)
                normal = cuda.local.array(3, dtype=np.float64)
                assign_value_3d(res_color, 1)
                assign_value_3d(grad, 0)
                assign_3d(current_point, camera_location)
                getPixelDirection(i, j, width, height, direction)
                calc_direction(camera_location, direction, direction)
                direct_flag = False
                depth_start_ind = depth_inds[tid]
                total_depth = depth_inds[tid+1] - depth_start_ind
                rrFactor = 1.0
                collisions = 0
                differ_color = object_mat[differ_ind, 5:8]
                for depth in range(total_depth):
                    depth_ind = depth_start_ind + depth
                    # Russian Roulette
                    if depth >= rr_depth:
                        rrFactor = 1.0 / (1.0 - rr_stop_prob)

                    # Scene Intersection
                    mint = calc_direction(current_point, hit_points[0, :, depth_ind], direction)
                    assign_3d(current_point, hit_points[0, :, depth_ind])
                    obj_ind = hit_objects[0, depth_ind]
                    diffusive = hit_objects[1, depth_ind]
                    # if ray didn't intersect anything
                    if obj_ind == -1:
                        print("bug")
                        return

                    # getting object properties
                    calc_normal(current_point, object_mat[obj_ind], normal)
                    emission = object_mat[obj_ind, 8]
                    obj_color = object_mat[obj_ind, 5:8]
                    # if we have reached the light source, add the contribution and terminate
                    if emission > 0:
                        if not direct_illumination or not direct_flag:
                            multiply_by_scalar(res_color, emission * rrFactor)
                        break

                    obj_Ks = object_mat[obj_ind, 9]  # specular constant
                    obj_n = object_mat[obj_ind, 10]  # glossiness constant
                    calc_specular_direction(direction, normal, specular_direction) # calculate specular direction

                    multiply_by_scalar(res_color, rrFactor)
                    if diffusive:  # Diffusive reflection
                        # adding the contribution of the object color (cosine term was canceled by MC)
                        if obj_ind == differ_ind:
                            multiply_by_scalar(res_color, ((1-obj_Ks)/(1-Ks0)))
                        res_color[0] *= obj_color[0]
                        res_color[1] *= obj_color[1]
                        res_color[2] *= obj_color[2]
                        # count collisions with the request object
                        if obj_ind == differ_ind:
                            collisions += 1

                    elif obj_ind==differ_ind:
                        multiply_by_scalar(res_color, (obj_Ks / Ks0))

                    # direct ilumination
                    assign_3d(temp_point, hit_points[1, :, depth_ind])
                    direct_flag = diffusive or (obj_n != -1 and obj_n < max_n_direct)
                    if direct_illumination and direct_flag and (temp_point[0] != 0 or temp_point[1] != 0 or temp_point[2] != 0):
                        distance = calc_direction(current_point, temp_point, direction)
                        mint = intersect_sphere(object_mat[-1, 1:4], object_mat[-1, 4], current_point, direction)
                        temp_point[0] = current_point[0] + mint * direction[0]
                        temp_point[1] = current_point[1] + mint * direction[1]
                        temp_point[2] = current_point[2] + mint * direction[2]
                        cost = dot_3d(direction, normal)  # surface cosine
                        calc_normal(temp_point, object_mat[-1], normal)  # using normal as temp for light normal
                        cosl = -dot_3d(direction, normal)  # light source cosine
                        # emission * r^-2 * cos(diffuse_angle) * cos(light_angle) * brdf * 1/pdf
                        temp = object_mat[-1, 8] * (1 / (distance * distance)) * cosl
                        rad_squared = object_mat[-1, 4] * object_mat[-1, 4]
                        if diffusive:
                            temp *= 2 * rad_squared * cost
                        else:
                            cosa = dot_3d(direction, specular_direction)
                            brdf = calc_specular_brdf(cosa, obj_n)
                            temp *= brdf * 2 * np.pi * rad_squared
                        grad[0] += res_color[0] * temp * collisions / differ_color[0]
                        grad[1] += res_color[1] * temp * collisions / differ_color[1]
                        grad[2] += res_color[2] * temp * collisions / differ_color[2]


                    # # Recycling Correction Factor
                    if obj_ind == differ_ind and depth < total_depth -1 and not diffusive:
                        calc_direction(current_point, hit_points[0, :, depth_ind + 1], direction)  # using rotY as temp
                        cosa = dot_3d(direction, specular_direction)
                        brdf_specular = calc_specular_brdf(cosa, obj_n)
                        brdf_specular0 = calc_specular_brdf(cosa, n0)
                        if obj_n == -1:
                            multiply_by_scalar(res_color, cosa>0.99)
                        else:
                            multiply_by_scalar(res_color, brdf_specular/brdf_specular0)

                # atomically adding the final contribution of the ray to the final image
                grad[0] += res_color[0] * collisions / differ_color[0]
                grad[1] += res_color[1] * collisions / differ_color[1]
                grad[2] += res_color[2] * collisions / differ_color[2]
                cuda.atomic.add(total_grad, 0, grad[0] * I_diff[j,i,0])
                cuda.atomic.add(total_grad, 1, grad[1] * I_diff[j,i,1])
                cuda.atomic.add(total_grad, 2, grad[2] * I_diff[j,i,2])

        @cuda.jit()
        def render_inverse_specular_cuda(job_inds, Ks0, n0, hit_points, hit_objects, depth_inds, object_mat, camera_location, width,
                                      height, spp, rng_states, I_diff, total_grad):
            tid = cuda.grid(1)
            if tid < job_inds.shape[0]:
                j,i = job_inds[tid]
                # Local arrays
                res_color = cuda.local.array(3, dtype=np.float64)
                grad = cuda.local.array(2, dtype=np.float64)
                grad_temp = cuda.local.array(2, dtype=np.float64)
                current_point = cuda.local.array(3, dtype=np.float64)
                direction = cuda.local.array(3, dtype=np.float64)
                temp_point = cuda.local.array(3, dtype=np.float64)
                specular_direction = cuda.local.array(3, dtype=np.float64)
                normal = cuda.local.array(3, dtype=np.float64)
                assign_value_3d(res_color, 1)
                grad[0] = 0
                grad[1] = 0
                grad_temp[0] = 0
                grad_temp[1] = 0

                direct_flag = False
                depth_start_ind = depth_inds[tid]
                total_depth = depth_inds[tid + 1] - depth_start_ind
                rrFactor = 1.0
                # init direction and location
                assign_3d(current_point, hit_points[0, :, depth_start_ind])
                calc_direction(camera_location, current_point, direction)
                for depth in range(total_depth):
                    depth_ind = depth_start_ind + depth
                    # Russian Roulette
                    if depth >= rr_depth:
                        rrFactor = 1.0 / (1.0 - rr_stop_prob)

                    # Scene Intersection
                    obj_ind = hit_objects[0, depth_ind]
                    diffusive = hit_objects[1, depth_ind]
                    # if ray didn't intersect anything
                    if obj_ind == -1:
                        print("bug")
                        return

                    # getting object properties
                    calc_normal(current_point, object_mat[obj_ind], normal)
                    emission = object_mat[obj_ind, 8]
                    obj_color = object_mat[obj_ind, 5:8]
                    # if we have reached the light source, add the contribution and terminate
                    if emission > 0:
                        if not direct_illumination or not direct_flag:
                            multiply_by_scalar(res_color, emission * rrFactor)
                        break

                    obj_ind_temp = obj_ind
                    obj_Ks = object_mat[obj_ind, 9]  # specular constant
                    obj_n = object_mat[obj_ind, 10]  # glossiness constant
                    calc_specular_direction(direction, normal, specular_direction)  # calculate specular direction

                    multiply_by_scalar(res_color, rrFactor)
                    if diffusive:  # Diffusive reflection
                        # adding the contribution of the object color (cosine term was canceled by MC)
                        if obj_ind == differ_ind:
                            multiply_by_scalar(res_color, ((1 - obj_Ks) / (1 - Ks0)))
                        res_color[0] *= obj_color[0]
                        res_color[1] *= obj_color[1]
                        res_color[2] *= obj_color[2]


                    elif obj_ind == differ_ind:
                        multiply_by_scalar(res_color, (obj_Ks / Ks0))

                    # direct ilumination
                    assign_3d(temp_point, hit_points[1, :, depth_ind])
                    direct_flag = diffusive or (obj_n != -1 and obj_n < max_n_direct)
                    if direct_illumination and direct_flag and (
                            temp_point[0] != 0 or temp_point[1] != 0 or temp_point[2] != 0):
                        distance = calc_direction(current_point, temp_point, direction)
                        mint = intersect_sphere(object_mat[-1, 1:4], object_mat[-1, 4], current_point, direction)
                        temp_point[0] = current_point[0] + mint * direction[0]
                        temp_point[1] = current_point[1] + mint * direction[1]
                        temp_point[2] = current_point[2] + mint * direction[2]
                        cost = dot_3d(direction, normal)  # surface cosine
                        calc_normal(temp_point, object_mat[-1], normal)  # using normal as temp for light normal
                        cosl = -dot_3d(direction, normal)  # light source cosine
                        # emission * r^-2 * cos(diffuse_angle) * cos(light_angle) * brdf * 1/pdf
                        temp = object_mat[-1, 8] * (1 / (distance * distance)) * cosl
                        rad_squared = object_mat[-1, 4] * object_mat[-1, 4]

                        cosa = dot_3d(direction, specular_direction)
                        brdf_specular = calc_specular_brdf(cosa, obj_n)
                        if diffusive:
                            temp *= 2 * rad_squared * cost
                        else:
                            temp *= brdf_specular * 2 * np.pi * rad_squared
                        if obj_ind == differ_ind:
                            # brdf = obj_Ks * brdf_specular + (1 - obj_Ks) * PI_INV
                            brdf = obj_Ks * brdf_specular + object_mat[differ_ind, 5] * PI_INV
                            if cosa > 0:
                                # n gradient
                                grad_direct = (obj_Ks * brdf_specular / brdf) * ((1 / (obj_n + 1)) + math.log(cosa))
                                grad_direct += grad_temp[0]
                                grad[0] += res_color[0] * temp * grad_direct * I_diff[j, i, 0]
                                grad[0] += res_color[1] * temp * grad_direct * I_diff[j, i, 1]
                                grad[0] += res_color[2] * temp * grad_direct * I_diff[j, i, 2]
                                # Ks gradient
                                # grad_direct = (brdf_specular - PI_INV) / brdf
                                grad_direct = brdf_specular / brdf
                            else:
                                # grad_direct = -(1 / (1 - obj_Ks))
                                grad_direct = 0
                            # Ks gradient continued
                            grad_direct += grad_temp[1]
                            grad[1] += res_color[0] * temp * grad_direct * I_diff[j, i, 0]
                            grad[1] += res_color[1] * temp * grad_direct * I_diff[j, i, 1]
                            grad[1] += res_color[2] * temp * grad_direct * I_diff[j, i, 2]


                    if depth < total_depth - 1:
                        calc_direction(current_point, hit_points[0, :, depth_ind + 1], direction)
                        assign_3d(current_point,  hit_points[0, :, depth_ind + 1])
                        if obj_ind == differ_ind:
                            # calc reflection gradient
                            cosa = dot_3d(direction, specular_direction)
                            brdf_specular = calc_specular_brdf(cosa, obj_n)
                            brdf = obj_Ks * brdf_specular + object_mat[differ_ind, 5] * PI_INV
                            if cosa > 0:
                                grad_temp[0] += (obj_Ks * brdf_specular / brdf) * (
                                        (1 / (obj_n + 1)) + math.log(cosa))

                                # grad_temp[1] += (brdf_specular - PI_INV) / brdf
                                grad_temp[1] += brdf_specular / brdf
                            # else:
                            #     grad_temp[1] += -(1 / (1 - obj_Ks))

                            if not diffusive:
                                # Recycling Correction Factor
                                cosa = dot_3d(direction, specular_direction)
                                brdf_specular = calc_specular_brdf(cosa, obj_n)
                                brdf_specular0 = calc_specular_brdf(cosa, n0)
                                if obj_n == -1:
                                    multiply_by_scalar(res_color, cosa > 0.99)
                                else:
                                    multiply_by_scalar(res_color, brdf_specular / brdf_specular0)



                # add gradient when the last ray did not perform direct illumination (obj_n exceeds max_direct_n)
                if emission > 0 and not direct_flag and depth > 0 and obj_ind_temp == obj_ind:
                    grad[0] += res_color[0] * grad_temp[0] * I_diff[j, i, 0]
                    grad[0] += res_color[1] * grad_temp[0] * I_diff[j, i, 1]
                    grad[0] += res_color[2] * grad_temp[0] * I_diff[j, i, 2]
                    grad[1] += res_color[0] * grad_direct * I_diff[j, i, 0]
                    grad[1] += res_color[1] * grad_direct * I_diff[j, i, 1]
                    grad[1] += res_color[2] * grad_direct * I_diff[j, i, 2]

                cuda.atomic.add(total_grad, 0, grad[0])
                cuda.atomic.add(total_grad, 1, grad[1])

        self.calc_total_reflection = calc_total_reflection
        self.generate_paths_cuda = generate_paths_cuda
        self.render_cuda = render_cuda
        self.render_inverse_color_cuda = render_inverse_color_cuda
        self.render_inverse_specular_cuda = render_inverse_specular_cuda
    def compile(self, spp, inverse_type):
        print("COMPILING")
        start = time()
        self.generate_paths(spp, init_rng=self.rng_states==None)
        self.render(0, inverse_type=inverse_type)
        print("Compilation took:", time()- start)

    def objects_to_device(self):
        del(self.dobject_mat)
        object_mat = np.empty((len(self.objects), 11))
        for i, obj in enumerate(self.objects):
            object_mat[i, 0] = obj.shape
            if obj.shape == 0:
                object_mat[i, 1:4] = obj.N
                object_mat[i, 4] = obj.d
            else:
                object_mat[i, 1:4] = obj.center
                object_mat[i, 4] = obj.radius
            object_mat[i, 5:8] = obj.color
            object_mat[i, 8] = obj.emission
            object_mat[i, 9] = obj.Ks
            object_mat[i, 10] = obj.n
        self.dobject_mat = cuda.to_device(object_mat)

    def generate_paths(self, spp, init_rng=True, to_sort=True):
        del(self.ddepth_inds)
        del(self.dhit_points)
        del(self.dhit_objects)
        grid_shape = (self.camera.width, self.camera.height, spp)
        Np = int(np.prod(grid_shape))
        if Np != self.Np:
            self.Np = Np
            self.job_inds = np.zeros((Np, 2), dtype=np.uint8)
            counter = 0
            for j in range(self.camera.height):
                for i in range(self.camera.width):
                    self.job_inds[counter:counter + spp, 0] = j
                    self.job_inds[counter:counter + spp, 1] = i
                    counter += spp
            self.djob_inds = cuda.device_array(self.job_inds.shape, dtype=np.uint8)

        self.djob_inds.copy_to_device(self.job_inds)

        self.Ks0 = self.objects[differ_ind].Ks
        self.n0 = self.objects[differ_ind].n
        self.objects_to_device()
        self.spp = spp

        depth_sizes = np.zeros(Np, dtype=np.uint16)
        ddepth_sizes = cuda.to_device(depth_sizes)
        print(f"Np = {Np:.2e}")
        seed = np.random.randint(1, int(1e9))
        if init_rng:
            print("Initialize RNG_STATES")
            start = time()
            del(self.rng_states)
            self.rng_states = create_xoroshiro128p_states(Np, seed)
            print("init rng took:",time()-start)
        rng_states_cpu = self.rng_states.copy_to_host()
        # threadsperblock = (8, 8, 8)
        self.threadsperblock = 256
        self.blockspergrid = (Np + (self.threadsperblock - 1)) // self.threadsperblock
        self.calc_total_reflection[self.blockspergrid, self.threadsperblock](self.djob_inds, self.dobject_mat, self.dcamera_location, self.camera.width,
                                                         self.camera.height, spp, self.rng_states, ddepth_sizes)
        cuda.synchronize()
        depth_sizes = ddepth_sizes.copy_to_host()
        del(ddepth_sizes)
        if to_sort:
            start = time()
            sort_inds = np.argsort(depth_sizes)
            print("sorting took:", time()-start)
        else:
            sort_inds = np.arange(depth_sizes.shape[0])
        self.djob_inds.copy_to_device(self.job_inds[sort_inds])
        depth_sizes = depth_sizes[sort_inds]
        rng_states_cpu = rng_states_cpu[sort_inds]
        total_depth = np.sum(depth_sizes)
        depth_inds = np.concatenate([np.array([0]), depth_sizes])
        depth_inds = np.cumsum(depth_inds)
        self.ddepth_inds = cuda.to_device(depth_inds)

        self.dhit_points = cuda.to_device(np.zeros((2, 3, total_depth), np.float64))
        self.dhit_objects = cuda.to_device(np.zeros((2, total_depth), np.uint8))
        del(self.rng_states)
        self.rng_states = cuda.to_device(rng_states_cpu)
        # print(f"paths weigth: {(self.dhit_points.nbytes + self.dhit_objects.nbytes + self.ddepth_inds.nbytes + self.rng_states.nbytes)/1e9} GB")
        # print(f"hitpoints weights: {self.dhit_points.nbytes/1e9} GB")
        self.generate_paths_cuda[self.blockspergrid, self.threadsperblock](self.djob_inds, self.dobject_mat, self.dcamera_location,
                                                                 self.camera.width,
                                                                 self.camera.height, spp, self.rng_states,
                                                                 self.ddepth_inds, self.dhit_points, self.dhit_objects)
        cuda.synchronize()
        return

    def render(self, I_gt=None, inverse_type=None):
        self.objects_to_device()
        I_res = np.zeros((self.camera.height, self.camera.width, 3), dtype=np.float64)
        dI_res = cuda.to_device(I_res)
        Np = int(I_res.shape[0]*I_res.shape[1]*self.spp)
        print(f"Np = {Np:.2e}")

        # threadsperblock = (8, 8, 8)
        # threadsperblock = (6, 6, 6)
        # blockspergrid_x = (self.camera.width + threadsperblock[0] -1) // threadsperblock[0]
        # blockspergrid_y = (self.camera.height + threadsperblock[1] -1) // threadsperblock[1]
        # blockspergrid_z = (self.spp + threadsperblock[2] -1) // threadsperblock[2]
        # blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
        start = time()
        self.render_cuda[self.blockspergrid, self.threadsperblock](self.djob_inds, self.Ks0, self.n0, self.dhit_points, self.dhit_objects, self.ddepth_inds,
                                                         self.dobject_mat, self.dcamera_location, self.camera.width,
                                                         self.camera.height, self.spp, dI_res)
        cuda.synchronize()
        print("render_cuda: ",time()-start)

        I_res = dI_res.copy_to_host()/self.spp
        del(dI_res)
        if I_gt is None:
            return I_res

        # differentiable part
        I_diff = I_res - I_gt
        dI_diff = cuda.to_device(I_diff)

        if inverse_type == "color":

            dtotal_grad = cuda.to_device(np.zeros(3, dtype=np.float64))
            start = time()
            inverse_func = self.render_inverse_color_cuda[self.blockspergrid, self.threadsperblock]
        elif inverse_type == "specular":
            dtotal_grad = cuda.to_device(np.zeros(2, dtype=np.float64))
            inverse_func = self.render_inverse_specular_cuda[self.blockspergrid, self.threadsperblock]
        else:
            print("inverse type is not supported")
            exit(1)
        inverse_func(self.djob_inds, self.Ks0, self.n0, self.dhit_points, self.dhit_objects, self.ddepth_inds,
                                                         self.dobject_mat, self.dcamera_location, self.camera.width,
                                                         self.camera.height, self.spp, self.rng_states, dI_diff, dtotal_grad)

        total_grad = dtotal_grad.copy_to_host()/self.spp
        print("inverse_func: ", time() - start)
        del(dtotal_grad)
        del(dI_diff)
        return I_res,  total_grad



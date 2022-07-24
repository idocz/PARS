from utils import *
from numba.cuda.random import create_xoroshiro128p_states, init_xoroshiro128p_states_cpu, xoroshiro128p_dtype
from numba import cuda
from time import time
class SceneGGX(object):
    def __init__(self, objects, albedo_image, plane_corners, plane_image_index, camera):
        self.objects = objects
        self.camera = camera
        self.albedo_image = albedo_image
        self.plane_corners = plane_corners
        self.plane_image_index = plane_image_index
        self.init = False
        self.rng_states_updated = None

        self.dobject_mat = None
        # camera_location = np.zeros(3, dtype=np.float)
        # camera_location[1] = 0.15

        # camera_location[2] -=0.5
        self.dcamera_location = cuda.to_device(camera.location)

        self.rng_states = None
        self.dalbedo_image = cuda.to_device(albedo_image)
        self.dplane_corners = cuda.to_device(plane_corners)
        self.ddepth_inds = None
        self.dhit_objects = None
        self.dsort_inds = None
        self.dgrad_contrib = None

        self.dI_res = cuda.device_array((self.camera.height, self.camera.width, 3), dtype=np.float64)
        self.dtotal_grad = cuda.device_array(albedo_image.shape, dtype=np.float64)

        @cuda.jit()
        def calc_total_reflection(object_mat, camera_location, width, height, spp, rng_states, depth_sizes):
            # i, j, s = cuda.grid(3)
            # tid = height * width * s + width * j + i
            # if i < width and j < height and s < spp:
            tid = cuda.grid(1)
            i, j, _ = transform_1D_to_3D(tid, width, height, spp)
            if tid < depth_sizes.shape[0]:
                current_point = cuda.local.array(3, dtype=np.float64)
                direction = cuda.local.array(3, dtype=np.float64)
                specular_direction = cuda.local.array(3, dtype=np.float64)
                normal = cuda.local.array(3, dtype=np.float64)
                sampleDir = cuda.local.array(3, dtype=np.float64)
                rotX = cuda.local.array(3, dtype=np.float64)
                rotY = cuda.local.array(3, dtype=np.float64)
                assign_3d(current_point, camera_location)
                getPixelDirection(i, j, width, height, direction)
                # calc_direction(camera_location, direction, direction)
                depth = 0
                # antialiasing
                current_point[0] += (2.0 * sample_unifrom(rng_states, tid) - 1.0) / 300
                current_point[1] += (2.0 * sample_unifrom(rng_states, tid) - 1.0) / 300
                calc_direction(current_point, direction, direction)
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

                    # increasing the ray depth
                    depth += 1

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
                                while True:
                                    ons_cuda(specular_direction, rotX, rotY)
                                    sample_hemisphere_cuda(obj_n, rng_states, tid, sampleDir)
                                    change_coordinate_system(sampleDir, specular_direction, rotX, rotY, direction)
                                    cost = dot_3d(normal, direction)
                                    if cost >= 0:
                                        break

                            else:  # Perfect specular
                                assign_3d(direction, specular_direction)

                    if diffusive:  # Diffusive reflection
                        # indirect illumination
                        ons_cuda(normal, rotX, rotY)
                        sample_hemisphere_cuda(1, rng_states, tid, sampleDir) # importance sampling the cosine term
                        change_coordinate_system(sampleDir, normal, rotX, rotY, direction)  # rotating hemisphere vector

                    # direct illumination (just sampling for rng_states sync)
                    direct_flag = diffusive or (obj_n != -1 and obj_n < max_n_direct)
                    if direct_illumination and direct_flag:
                        sample_sphere(object_mat[-1, 1:5], rng_states, tid, rotX)  # using temp rotX for light location

                # track depth for each path
                depth_sizes[tid] = depth

        @cuda.jit()
        def generate_paths_cuda(object_mat, camera_location, width, height, spp, rng_states, depth_inds,
                                hit_objects, sort_inds):
            # i, j, s = cuda.grid(3)
            # if i < width and j < height and s < spp:
            # old_tid = height * width * s + width * j + i
            old_tid = cuda.grid(1)
            if old_tid < sort_inds.shape[0]:
                tid = sort_inds[old_tid]
                i, j, _ = transform_1D_to_3D(tid, width, height, spp)


                # print(tid, i, j, s)
                current_point = cuda.local.array(3, dtype=np.float64)
                direction = cuda.local.array(3, dtype=np.float64)
                specular_direction = cuda.local.array(3, dtype=np.float64)
                normal = cuda.local.array(3, dtype=np.float64)
                sampleDir = cuda.local.array(3, dtype=np.float64)
                rotX = cuda.local.array(3, dtype=np.float64)
                rotY = cuda.local.array(3, dtype=np.float64)
                assign_3d(current_point, camera_location)
                getPixelDirection(i, j, width, height, direction)
                current_point[0] += (2.0*sample_unifrom(rng_states, tid)-1.0)/300
                current_point[1] += (2.0*sample_unifrom(rng_states, tid)-1.0)/300
                calc_direction(current_point, direction, direction)
                depth_start_ind = depth_inds[old_tid]
                # total_depth = depth_inds[tid + 1] - depth_start_ind
                depth = 0
                while True:
                    depth_ind = depth_start_ind + depth

                    # Russian Roulette
                    if depth >= rr_depth:
                        if sample_unifrom(rng_states, tid) <= rr_stop_prob:
                            # if total_depth != depth:
                            #     print("bug total_depth != depth")
                            break

                    # Scene Intersection
                    obj_ind, mint = intersect_all(object_mat, current_point, direction)

                    # if ray didn't intersect anything (does not suppose to happened)
                    if obj_ind == -1:
                        print("bug obj_ind ==-1")
                        return

                    # stepping towards the intersected object
                    step_in_direction(current_point, direction, mint)

                    # saving path point and object
                    hit_objects[depth_ind] = obj_ind

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
                                while True:
                                    ons_cuda(specular_direction, rotX, rotY)
                                    sample_hemisphere_cuda(obj_n, rng_states, tid, sampleDir)
                                    change_coordinate_system(sampleDir, specular_direction, rotX, rotY, direction)
                                    cost = dot_3d(normal, direction)
                                    if cost >= 0:
                                        break

                            else:  # Perfect specular
                                assign_3d(direction, specular_direction)

                    if diffusive:  # Diffusive reflection
                        # indirect illumination
                        ons_cuda(normal, rotX, rotY)
                        sample_hemisphere_cuda(1, rng_states, tid, sampleDir)  # importance sampling the cosine term
                        change_coordinate_system(sampleDir, normal, rotX, rotY, direction)  # rotating hemisphere vector

                    # direct illumination
                    direct_flag = diffusive or (obj_n != -1 and obj_n < max_n_direct)
                    if direct_illumination and direct_flag:
                        sample_sphere(object_mat[-1, 1:5], rng_states, tid, rotX)  # using temp rotX for light location
                        # distance = calc_direction(current_point, rotX, rotY)  # using rotY for light direction
                        # direct_ind, mint = intersect_all(object_mat, current_point, rotY)
                        # direct_ind, mint = 9, 1
                        # cosl = 0
                        # cost = 0
                        # if direct_ind == object_mat.shape[0] - 1:  # light source visibility test
                        #     rotX[0] = current_point[0] + mint * rotY[0]
                        #     rotX[1] = current_point[1] + mint * rotY[1]
                        #     rotX[2] = current_point[2] + mint * rotY[2]
                        #     cost = dot_3d(rotY, normal)  # surface cosine
                        #     calc_normal(rotX, object_mat[direct_ind], normal)
                        #     cosl = -dot_3d(rotY, normal)  # light source cosine

                    # increasing the ray depth
                    depth += 1

        @cuda.jit()
        def render_recycling_cuda(hit_objects, depth_inds, object_mat, albedo_image, plane_corners, plane_image_index, camera_location, width, height, spp,
                        rng_states, I_res, sort_inds, grad_contrib):
            # i, j, s = cuda.grid(3)
            # if i < width and j < height and s < spp:
            # old_tid = height * width * s + width * j + i
            old_tid = cuda.grid(1)
            if old_tid < sort_inds.shape[0]:

                tid = sort_inds[old_tid]
                i, j, _ = transform_1D_to_3D(tid, width, height, spp)
                depth = 0
                # Local arrays
                res_color = cuda.local.array(3, dtype=np.float64)
                total_color = cuda.local.array(3, dtype=np.float64)
                current_point = cuda.local.array(3, dtype=np.float64)
                direction = cuda.local.array(3, dtype=np.float64)
                specular_direction = cuda.local.array(3, dtype=np.float64)
                normal = cuda.local.array(3, dtype=np.float64)
                sampleDir = cuda.local.array(3, dtype=np.float64)
                rotX = cuda.local.array(3, dtype=np.float64)
                rotY = cuda.local.array(3, dtype=np.float64)
                obj_color = cuda.local.array(3, dtype=np.float32)
                assign_value_3d(res_color, 1)
                assign_value_3d(total_color, 0)
                assign_3d(current_point, camera_location)
                getPixelDirection(i, j, width, height, direction)
                # antialiasing
                current_point[0] += (2.0 * sample_unifrom(rng_states, tid) - 1.0) / 300
                current_point[1] += (2.0 * sample_unifrom(rng_states, tid) - 1.0) / 300
                calc_direction(current_point, direction, direction)
                direct_flag = False
                depth_start_ind = depth_inds[old_tid]
                total_depth = depth_inds[old_tid + 1] - depth_start_ind
                while True:
                    depth_ind = depth_start_ind + depth
                    # if depth >= total_depth:
                    #     print(depth, total_depth)
                    rrFactor = 1.0
                    # Russian Roulette
                    if depth >= rr_depth:
                        if sample_unifrom(rng_states, tid) <= rr_stop_prob:
                            assign_value_3d(res_color, 0)
                            break

                        rrFactor = 1.0 / (1.0 - rr_stop_prob)

                    grad_contrib[0,depth_ind] = 0
                    grad_contrib[1,depth_ind] = 0
                    grad_contrib[2,depth_ind] = 0
                    # Scene Intersection
                    obj_ind = hit_objects[depth_ind]
                    mint = intersect(object_mat[obj_ind, :5], current_point, direction)

                    # if ray didn't intersect anything
                    if obj_ind == -1:
                        return

                    # stepping towards the intersected object
                    step_in_direction(current_point, direction, mint)

                    # getting object properties
                    calc_normal(current_point, object_mat[obj_ind], normal)
                    emission = object_mat[obj_ind, 8]
                    assign_3d(obj_color, object_mat[obj_ind, 5:8])
                    # if we have reached the light source, add the contribution and terminate
                    if emission > 0:
                        if not direct_illumination or not direct_flag:
                            multiply_by_scalar(res_color, emission * rrFactor)
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
                                while True:
                                    ons_cuda(specular_direction, rotX, rotY)
                                    sample_hemisphere_cuda(obj_n, rng_states, tid, sampleDir)
                                    change_coordinate_system(sampleDir, specular_direction, rotX, rotY, direction)
                                    cost = dot_3d(normal, direction)
                                    if cost >= 0:
                                        multiply_by_scalar(res_color, rrFactor)
                                        break

                            else:  # Perfect specular
                                assign_3d(direction, specular_direction)
                                multiply_by_scalar(res_color, rrFactor)

                    if diffusive:  # Diffusive reflection
                        # adding the contribution of the object color (cosine term was canceled by MC)
                        if obj_ind == plane_image_index:
                            get_pixel_albedo(current_point, albedo_image, plane_corners, obj_color)
                        res_color[0] *= obj_color[0] * rrFactor
                        res_color[1] *= obj_color[1] * rrFactor
                        res_color[2] *= obj_color[2] * rrFactor

                        # indirect illumination
                        ons_cuda(normal, rotX, rotY)
                        sample_hemisphere_cuda(1, rng_states, tid, sampleDir)  # importance sampling the cosine term
                        change_coordinate_system(sampleDir, normal, rotX, rotY, direction)  # rotating hemisphere vector

                    # direct ilumination
                    direct_flag = diffusive or (obj_n != -1 and obj_n < max_n_direct)
                    if direct_illumination and direct_flag:
                        sample_sphere(object_mat[-1, 1:5], rng_states, tid,
                                      rotX)  # using rotX as temp for light location
                        distance = calc_direction(current_point, rotX, rotY)  # Using rotY as temp for light direction
                        direct_ind, mint = intersect_all(object_mat, current_point, rotY)
                        if direct_ind == object_mat.shape[0] - 1:  # light source visibility test
                            rotX[0] = current_point[0] + mint * rotY[0]
                            rotX[1] = current_point[1] + mint * rotY[1]
                            rotX[2] = current_point[2] + mint * rotY[2]
                            calc_normal(rotX, object_mat[direct_ind],
                                        sampleDir)  # using sample_dir as temp for light normal
                            cosl = -dot_3d(rotY, sampleDir)  # light source cosine
                            cost = dot_3d(rotY, normal)  # surface cosine
                            if cosl > 0 and cost > 0:
                                # emission * r^-2 * cos(diffuse_angle) * cos(light_angle) * brdf * 1/pdf
                                temp = object_mat[-1, 8] * (1 / (distance * distance)) * cosl
                                rad_squared = object_mat[-1, 4] * object_mat[-1, 4]
                                if diffusive:
                                    temp *= 2 * rad_squared * cost
                                else:
                                    cosa = dot_3d(rotY, specular_direction)
                                    brdf = calc_specular_brdf(cosa, obj_n)
                                    temp *= brdf * 2 * np.pi * rad_squared

                                for color_ind in range(3):
                                    loop_temp = res_color[color_ind] * temp
                                    total_color[color_ind] += loop_temp
                                    grad_contrib[color_ind, depth_ind] = loop_temp
                    # increasing the ray depth
                    depth += 1

                # atomically adding the final contribution of the ray to the final image
                if direct_illumination:
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

                grad_sum_R = 0
                grad_sum_G = 0
                grad_sum_B = 0
                for depth in range(total_depth):
                    depth_ind = total_depth - 1 - depth + depth_start_ind
                    grad_sum_R += grad_contrib[0, depth_ind]
                    grad_sum_G += grad_contrib[1, depth_ind]
                    grad_sum_B += grad_contrib[2, depth_ind]
                    grad_contrib[0, depth_ind] = grad_sum_R
                    grad_contrib[1, depth_ind] = grad_sum_G
                    grad_contrib[2, depth_ind] = grad_sum_B

        @cuda.jit()
        def render_differentiable_recycling_cuda(hit_objects, depth_inds, object_mat, albedo_image, plane_corners, plane_image_index,
                                  camera_location, width, height, spp,
                                  rng_states, I_dif, sort_inds, grad_contrib, total_grad):
            old_tid = cuda.grid(1)
            if old_tid < sort_inds.shape[0]:

                tid = sort_inds[old_tid]
                i, j, _ = transform_1D_to_3D(tid, width, height, spp)
                s = tid // (height * width)
                temp = tid - (s * height * width)
                j = temp // width
                i = temp % width
                depth = 0
                # Local arrays
                res_color = cuda.local.array(3, dtype=np.float64)
                total_color = cuda.local.array(3, dtype=np.float64)
                current_point = cuda.local.array(3, dtype=np.float64)
                direction = cuda.local.array(3, dtype=np.float64)
                specular_direction = cuda.local.array(3, dtype=np.float64)
                normal = cuda.local.array(3, dtype=np.float64)
                sampleDir = cuda.local.array(3, dtype=np.float64)
                rotX = cuda.local.array(3, dtype=np.float64)
                rotY = cuda.local.array(3, dtype=np.float64)
                obj_color = cuda.local.array(3, dtype=np.float32)
                assign_value_3d(res_color, 1)
                assign_value_3d(total_color, 0)
                assign_3d(current_point, camera_location)
                getPixelDirection(i, j, width, height, direction)
                # antialiasing
                current_point[0] += (2.0 * sample_unifrom(rng_states, tid) - 1.0) / 300
                current_point[1] += (2.0 * sample_unifrom(rng_states, tid) - 1.0) / 300
                calc_direction(current_point, direction, direction)
                direct_flag = False
                depth_start_ind = depth_inds[old_tid]
                total_depth = depth_inds[old_tid + 1] - depth_start_ind
                while True:
                    albedo_i = -1
                    albedo_j = -1
                    depth_ind = depth_start_ind + depth
                    # rrFactor = 1.0
                    # Russian Roulette
                    if depth >= rr_depth:
                        if sample_unifrom(rng_states, tid) <= rr_stop_prob:
                            assign_value_3d(res_color, 0)
                            break

                        rrFactor = 1.0 / (1.0 - rr_stop_prob)

                    # Scene Intersection
                    obj_ind = hit_objects[depth_ind]
                    mint = intersect(object_mat[obj_ind, :5], current_point, direction)

                    # if ray didn't intersect anything
                    if obj_ind == -1:
                        return

                    # stepping towards the intersected object
                    step_in_direction(current_point, direction, mint)

                    # getting object properties
                    calc_normal(current_point, object_mat[obj_ind], normal)
                    emission = object_mat[obj_ind, 8]
                    assign_3d(obj_color, object_mat[obj_ind, 5:8])
                    # if we have reached the light source, add the contribution and terminate
                    if emission > 0:
                    #     if not direct_illumination or not direct_flag:
                    #         multiply_by_scalar(res_color, emission * rrFactor)
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
                                while True:
                                    ons_cuda(specular_direction, rotX, rotY)
                                    sample_hemisphere_cuda(obj_n, rng_states, tid, sampleDir)
                                    change_coordinate_system(sampleDir, specular_direction, rotX, rotY, direction)
                                    cost = dot_3d(normal, direction)
                                    if cost >= 0:
                                        multiply_by_scalar(res_color, rrFactor)
                                        break

                            else:  # Perfect specular
                                assign_3d(direction, specular_direction)
                                # multiply_by_scalar(res_color, rrFactor)

                    if diffusive:  # Diffusive reflection
                        # adding the contribution of the object color (cosine term was canceled by MC)
                        if obj_ind == plane_image_index:
                            albedo_i, albedo_j = get_pixel_albedo(current_point, albedo_image, plane_corners, obj_color)
                        # res_color[0] *= obj_color[0] * rrFactor
                        # res_color[1] *= obj_color[1] * rrFactor
                        # res_color[2] *= obj_color[2] * rrFactor

                        # indirect illumination
                        ons_cuda(normal, rotX, rotY)
                        sample_hemisphere_cuda(1, rng_states, tid, sampleDir)  # importance sampling the cosine term
                        change_coordinate_system(sampleDir, normal, rotX, rotY, direction)  # rotating hemisphere vector

                    # direct ilumination
                    direct_flag = diffusive or (obj_n != -1 and obj_n < max_n_direct)
                    if direct_illumination and direct_flag:
                        sample_sphere(object_mat[-1, 1:5], rng_states, tid,
                                      rotX)  # using rotX as temp for light location
                        # distance = calc_direction(current_point, rotX, rotY)  # Using rotY as temp for light direction
                        # direct_ind, mint = intersect_all(object_mat, current_point, rotY)
                        # if direct_ind == object_mat.shape[0] - 1:  # light source visibility test
                        #     rotX[0] = current_point[0] + mint * rotY[0]
                        #     rotX[1] = current_point[1] + mint * rotY[1]
                        #     rotX[2] = current_point[2] + mint * rotY[2]
                        #     calc_normal(rotX, object_mat[direct_ind],
                        #                 sampleDir)  # using sample_dir as temp for light normal
                            # cosl = -dot_3d(rotY, sampleDir)  # light source cosine
                            # cost = dot_3d(rotY, normal)  # surface cosine
                            # if cosl > 0 and cost > 0:
                                # emission * r^-2 * cos(diffuse_angle) * cos(light_angle) * brdf * 1/pdf
                                # temp = object_mat[-1, 8] * (1 / (distance * distance)) * cosl
                                # rad_squared = object_mat[-1, 4] * object_mat[-1, 4]
                                # if diffusive:
                                #     temp *= 2 * rad_squared * cost
                                # else:
                                #     cosa = dot_3d(rotY, specular_direction)
                                #     brdf = calc_specular_brdf(cosa, obj_n)
                                #     temp *= brdf * 2 * np.pi * rad_squared

                    if albedo_i != -1:
                        for color_ind in range(3):
                            grad_temp = I_dif[j,i,color_ind] * grad_contrib[color_ind, depth_ind] / obj_color[color_ind]
                            cuda.atomic.add(total_grad, (albedo_i, albedo_j,color_ind), grad_temp)


                    # increasing the ray depth
                    depth += 1

                    # print(total_depth,depth)
                # # atomically adding the final contribution of the ray to the final image
                # if direct_illumination:
                #     total_color[0] += res_color[0]
                #     total_color[1] += res_color[1]
                #     total_color[2] += res_color[2]
                #     cuda.atomic.add(I_res, (j, i, 0), total_color[0])
                #     cuda.atomic.add(I_res, (j, i, 1), total_color[1])
                #     cuda.atomic.add(I_res, (j, i, 2), total_color[2])
                # else:
                #     cuda.atomic.add(I_res, (j, i, 0), res_color[0])
                #     cuda.atomic.add(I_res, (j, i, 1), res_color[1])
                #     cuda.atomic.add(I_res, (j, i, 2), res_color[2])

        @cuda.jit()
        def render_cuda(object_mat,  albedo_image, plane_corners, plane_image_index, camera_location, width, height, spp, rng_states, I_res):
            # i, j, s = cuda.grid(3)
            # tid = height * width * s + width * j + i
            # if i < width and j < height and s < spp:
            tid = cuda.grid(1)
            if tid < rng_states.shape[0]:
                i, j, _ = transform_1D_to_3D(tid, width, height, spp)
                depth = 0
                # Local arrays
                res_color = cuda.local.array(3, dtype=np.float64)
                total_color = cuda.local.array(3, dtype=np.float64)
                current_point = cuda.local.array(3, dtype=np.float64)
                direction = cuda.local.array(3, dtype=np.float64)
                specular_direction = cuda.local.array(3, dtype=np.float64)
                normal = cuda.local.array(3, dtype=np.float64)
                sampleDir = cuda.local.array(3, dtype=np.float64)
                rotX = cuda.local.array(3, dtype=np.float64)
                rotY = cuda.local.array(3, dtype=np.float64)
                obj_color = cuda.local.array(3, dtype=np.float32)
                assign_value_3d(res_color, 1)
                assign_value_3d(total_color, 0)
                assign_3d(current_point, camera_location)
                getPixelDirection(i, j, width, height, direction)
                # antialiasing
                current_point[0] += (2.0*sample_unifrom(rng_states, tid)-1.0)/300
                current_point[1] += (2.0*sample_unifrom(rng_states, tid)-1.0)/300
                calc_direction(current_point, direction, direction)
                direct_flag = False
                while True:
                    rrFactor = 1.0
                    # Russian Roulette
                    if depth >= rr_depth:
                        if sample_unifrom(rng_states, tid) <= rr_stop_prob:
                            assign_value_3d(res_color, 0)
                            break

                        rrFactor = 1.0 / (1.0 - rr_stop_prob)

                    # Scene Intersection
                    obj_ind, mint = intersect_all(object_mat, current_point, direction)

                    # if ray didn't intersect anything
                    if obj_ind == -1:
                        return

                    # stepping towards the intersected object
                    step_in_direction(current_point, direction, mint)


                    # getting object properties
                    calc_normal(current_point, object_mat[obj_ind], normal)
                    emission = object_mat[obj_ind, 8]
                    assign_3d(obj_color,object_mat[obj_ind, 5:8])
                    # if we have reached the light source, add the contribution and terminate
                    if emission > 0 :
                        if not direct_illumination or not direct_flag:
                            multiply_by_scalar(res_color, emission * rrFactor)
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
                                while True:
                                    ons_cuda(specular_direction, rotX, rotY)
                                    sample_hemisphere_cuda(obj_n, rng_states, tid, sampleDir)
                                    change_coordinate_system(sampleDir, specular_direction, rotX, rotY, direction)
                                    cost = dot_3d(normal, direction)
                                    if cost >= 0:
                                        multiply_by_scalar(res_color, rrFactor)
                                        break

                            else:  # Perfect specular
                                assign_3d(direction, specular_direction)
                                multiply_by_scalar(res_color, rrFactor)

                    if diffusive:  # Diffusive reflection
                        # adding the contribution of the object color (cosine term was canceled by MC)
                        if obj_ind == plane_image_index:
                            get_pixel_albedo(current_point, albedo_image, plane_corners,obj_color)
                        res_color[0] *= obj_color[0] * rrFactor
                        res_color[1] *= obj_color[1] * rrFactor
                        res_color[2] *= obj_color[2] * rrFactor

                        # indirect illumination
                        ons_cuda(normal, rotX, rotY)
                        sample_hemisphere_cuda(1, rng_states, tid, sampleDir) # importance sampling the cosine term
                        change_coordinate_system(sampleDir, normal, rotX, rotY, direction)  # rotating hemisphere vector

                    # direct ilumination
                    direct_flag = diffusive or (obj_n!=-1 and obj_n < max_n_direct)
                    if direct_illumination and direct_flag:
                        sample_sphere(object_mat[-1, 1:5], rng_states, tid, rotX)  # using rotX as temp for light location
                        distance = calc_direction(current_point, rotX, rotY) # Using rotY as temp for light direction
                        direct_ind, mint = intersect_all(object_mat, current_point, rotY)
                        if direct_ind == object_mat.shape[0] - 1:  # light source visibility test
                            rotX[0] = current_point[0] + mint * rotY[0]
                            rotX[1] = current_point[1] + mint * rotY[1]
                            rotX[2] = current_point[2] + mint * rotY[2]
                            calc_normal(rotX, object_mat[direct_ind], sampleDir)  # using sample_dir as temp for light normal
                            cosl = -dot_3d(rotY, sampleDir) # light source cosine
                            cost = dot_3d(rotY, normal) # surface cosine
                            if cosl > 0 and cost > 0:
                                # emission * r^-2 * cos(diffuse_angle) * cos(light_angle) * brdf * 1/pdf
                                temp = object_mat[-1,8] * (1 / (distance * distance)) * cosl
                                rad_squared = object_mat[-1, 4] * object_mat[-1, 4]
                                if diffusive:
                                    temp *= 2 * rad_squared * cost
                                else:
                                    cosa = dot_3d(rotY, specular_direction)
                                    brdf = calc_specular_brdf(cosa, obj_n)
                                    temp *= brdf * 2 * np.pi * rad_squared

                                total_color[0] += res_color[0] * temp
                                total_color[1] += res_color[1] * temp
                                total_color[2] += res_color[2] * temp
                    # increasing the ray depth
                    depth += 1

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



        self.calc_total_reflection = calc_total_reflection
        self.generate_paths_cuda = generate_paths_cuda
        self.render_cuda = render_cuda
        self.render_recycling_cuda = render_recycling_cuda
        self.render_differentiable_recycling_cuda = render_differentiable_recycling_cuda


    def compile(self, spp, inverse_type):
        self.render(spp, 0, inverse_type, init_rng=self.rng_states==None)

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



    def init_cuda_param(self, spp, init=False, seed=None):
        # self.threadsperblock = (6, 6, 6)
        # blockspergrid_x = (self.camera.width + self.threadsperblock[0] - 1) // self.threadsperblock[0]
        # blockspergrid_y = (self.camera.height + self.threadsperblock[1] - 1) // self.threadsperblock[1]
        # blockspergrid_z = (spp + self.threadsperblock[2] - 1) // self.threadsperblock[2]
        # self.blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
        self.threadsperblock = 256
        self.blockspergrid = (self.Np + (self.threadsperblock - 1)) // self.threadsperblock
        if init:
            if seed is None:
                self.seed = np.random.randint(1, int(1e9))
            else:
                self.seed = seed
            self.rng_states = np.empty(shape=self.Np, dtype=xoroshiro128p_dtype)
            init_xoroshiro128p_states_cpu(self.rng_states, self.seed, 0)
            self.init = True
            self.rng_states_updated = None


    def generate_paths(self, spp, init_rng=True, to_sort=True, to_print=False):
        assert self.init or init_rng, "cuda rng was not initialized"
        if self.rng_states_updated is not None:
            self.rng_states = self.rng_states_updated
        del (self.ddepth_inds)
        del (self.dhit_objects)
        del(self.dsort_inds)
        del(self.dgrad_contrib)
        self.Ks0 = self.objects[differ_ind].Ks
        self.n0 = self.objects[differ_ind].n
        self.objects_to_device()
        self.spp = spp
        grid_shape = (self.camera.width, self.camera.height, spp)
        Np = int(np.prod(grid_shape))
        self.Np = Np
        depth_sizes = np.zeros(Np, dtype=np.uint8)
        ddepth_sizes = cuda.to_device(depth_sizes)
        if to_print:
            print(f"Np = {Np:.2e}")
            print("Initialize RNG_STATES")
        start = time()
        self.init_cuda_param(spp, init=init_rng)
        if to_print:
            print("init rng took:", time() - start)
        drng_states = cuda.to_device(self.rng_states)
        threadsperblock = self.threadsperblock
        blockspergrid = self.blockspergrid
        self.calc_total_reflection[blockspergrid, threadsperblock](self.dobject_mat, self.dcamera_location,
                                                                   self.camera.width,
                                                                   self.camera.height, spp, drng_states,
                                                                   ddepth_sizes)
        cuda.synchronize()
        depth_sizes = ddepth_sizes.copy_to_host()
        del (ddepth_sizes)
        self.rng_states_updated = drng_states.copy_to_host()
        del(drng_states)
        # plt.figure()
        # plt.hist(depth_sizes)
        # plt.show()
        if to_sort:
            sort_ind = np.argsort(depth_sizes)
        else:
            sort_ind = np.arange(depth_sizes.shape[0])
        self.dsort_inds = cuda.to_device(sort_ind)
        depth_sizes = depth_sizes[sort_ind]
        total_depth = np.sum(depth_sizes)
        depth_inds = np.concatenate([np.array([0]), depth_sizes])
        depth_inds = np.cumsum(depth_inds)
        self.ddepth_inds = cuda.to_device(depth_inds)

        self.dhit_objects = cuda.to_device(np.empty(total_depth, np.uint8))
        self.dgrad_contrib = cuda.to_device(np.empty((3,total_depth), np.float32))
        drng_states = cuda.to_device(self.rng_states)
        if to_print:
            print(
                f"paths weigth: {(+ self.dhit_objects.nbytes + self.ddepth_inds.nbytes + self.rng_states.nbytes) / 1e9} GB")
        self.generate_paths_cuda[blockspergrid, threadsperblock](self.dobject_mat, self.dcamera_location,
                                                                 self.camera.width,
                                                                 self.camera.height, spp, drng_states,
                                                                 self.ddepth_inds, self.dhit_objects, self.dsort_inds)
        cuda.synchronize()
        print("generated paths")
        return


    def render_recycling(self, I_gt=None, to_print=False):
        spp = self.spp
        # self.objects_to_device()
        self.dalbedo_image.copy_to_device(self.albedo_image)
        self.dI_res.copy_to_device(np.zeros((self.camera.height, self.camera.width, 3), dtype=np.float64))
        Np = int(self.dI_res.shape[0]*self.dI_res.shape[1]*spp)
        if to_print:
            print(f"Np = {Np:.2e}")
        # threadsperblock = (8, 8, 8)
        threadsperblock = self.threadsperblock
        blockspergrid = self.blockspergrid
        drng_states = cuda.to_device(self.rng_states)
        start = time()
        self.render_recycling_cuda[blockspergrid, threadsperblock](self.dhit_objects, self.ddepth_inds, self.dobject_mat, self.dalbedo_image, self.dplane_corners,
                                                         self.plane_image_index, self.dcamera_location, self.camera.width,
                                                         self.camera.height, spp, drng_states, self.dI_res, self.dsort_inds, self.dgrad_contrib)
        cuda.synchronize()
        if to_print:
            print("render_recycling_cuda:", time()-start)
        I_res = self.dI_res.copy_to_host()/spp
        if I_gt is None:
            return I_res

        self.dI_res.copy_to_device(I_res-I_gt)
        self.dtotal_grad.copy_to_device(np.zeros_like(self.albedo_image))
        drng_states = cuda.to_device(self.rng_states)
        self.render_differentiable_recycling_cuda[blockspergrid, threadsperblock]\
        (self.dhit_objects, self.ddepth_inds,
         self.dobject_mat, self.dalbedo_image,
         self.dplane_corners,
         self.plane_image_index, self.dcamera_location,
         self.camera.width,
         self.camera.height, spp, drng_states, self.dI_res,
         self.dsort_inds, self.dgrad_contrib, self.dtotal_grad)

        total_grad = self.dtotal_grad.copy_to_host()/self.spp
        del(drng_states)
        return I_res, total_grad

    def render(self, spp,  to_init_rng=False):
        self.objects_to_device()
        I_res = np.zeros((self.camera.height, self.camera.width, 3), dtype=np.float64)
        self.dI_res = cuda.to_device(I_res)
        Np = int(I_res.shape[0]*I_res.shape[1]*spp)
        self.Np = Np
        print(f"Np = {Np:.2e}")
        if to_init_rng:
            self.init_cuda_param(spp, init=to_init_rng)
        # threadsperblock = (8, 8, 8)

        threadsperblock = self.threadsperblock
        blockspergrid = self.blockspergrid
        start = time()
        self.render_cuda[blockspergrid, threadsperblock](self.dobject_mat, self.dalbedo_image, self.dplane_corners,
                                                         self.plane_image_index, self.dcamera_location, self.camera.width,
                                                         self.camera.height, spp, self.rng_states, self.dI_res)
        cuda.synchronize()
        print("render_cuda:", time()-start)
        I_res = self.dI_res.copy_to_host()/spp
        return I_res

    def update_albedo_image(self, albedo_image):
        self.albedo_image = albedo_image
        self.dalbedo_image.copy_to_device(albedo_image)


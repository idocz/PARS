from utils import *
from classes.ray import Ray
import multiprocessing as mp
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states
from time import time
class SceneGPU(object):
    def __init__(self, objects, camera):
        self.objects = objects
        self.camera = camera

        self.dobject_mat = None
        self.dcamera_location = cuda.to_device(np.zeros(3, dtype=np.float))
        self.rng_states = None

        @cuda.jit()
        def render_cuda(object_mat, camera_location, width, height, spp, rng_states, I_res):
            i, j, s = cuda.grid(3)
            tid = height * width * s + width * j + i
            if i < width and j < height and s < spp:
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
                assign_value_3d(res_color, 1)
                assign_value_3d(total_color, 0)
                assign_3d(current_point, camera_location)
                getPixelDirection(i, j, width, height, direction)
                # antialiasing
                # current_point[0] += (2.0*sample_unifrom(rng_states, tid)-1.0)/300
                # current_point[1] += (2.0*sample_unifrom(rng_states, tid)-1.0)/300
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
                    obj_color = object_mat[obj_ind, 5:8]
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

        @cuda.jit()
        def render_inverse_color_cuda(object_mat, camera_location, width, height, spp, rng_states, I_diff, total_grad):
            i, j, s = cuda.grid(3)
            tid = height * width * s + width * j + i
            if i < width and j < height and s < spp:
                depth = 0
                # Local arrays
                res_color = cuda.local.array(3, dtype=np.float64)
                grad = cuda.local.array(3, dtype=np.float64)
                current_point = cuda.local.array(3, dtype=np.float64)
                direction = cuda.local.array(3, dtype=np.float64)
                specular_direction = cuda.local.array(3, dtype=np.float64)
                normal = cuda.local.array(3, dtype=np.float64)
                sampleDir = cuda.local.array(3, dtype=np.float64)
                rotX = cuda.local.array(3, dtype=np.float64)
                rotY = cuda.local.array(3, dtype=np.float64)
                assign_value_3d(res_color, 1)
                assign_value_3d(grad, 0)
                assign_3d(current_point, camera_location)
                getPixelDirection(i, j, width, height, direction)
                calc_direction(camera_location, direction, direction)
                direct_flag = False
                differ_color = object_mat[differ_ind, 5:8]
                collisions = 0
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

                    # calc intesection normal
                    calc_normal(current_point, object_mat[obj_ind], normal)

                    # getting object properties
                    calc_normal(current_point, object_mat[obj_ind], normal)
                    emission = object_mat[obj_ind, 8]
                    obj_color = object_mat[obj_ind, 5:8]
                    # if we have reached the light source, add the contribution and terminate
                    if emission > 0:
                        if not direct_flag:
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
                        res_color[0] *= obj_color[0] * rrFactor
                        res_color[1] *= obj_color[1] * rrFactor
                        res_color[2] *= obj_color[2] * rrFactor

                        # indirect illumination
                        ons_cuda(normal, rotX, rotY)
                        sample_hemisphere_cuda(1, rng_states, tid, sampleDir)  # importance sampling the cosine term
                        change_coordinate_system(sampleDir, normal, rotX, rotY, direction)  # rotating hemisphere vector

                        # count collisions with the request object
                        if obj_ind == differ_ind:
                            collisions += 1
                    # direct ilumination
                    direct_flag = diffusive or (obj_n != -1 and obj_n < max_n_direct)
                    if direct_flag:
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
                                grad[0] +=  res_color[0] * temp * collisions / differ_color[0]
                                grad[1] +=  res_color[1] * temp * collisions / differ_color[1]
                                grad[2] +=  res_color[2] * temp * collisions / differ_color[2]
                    # increasing the ray depth
                    depth += 1

                # atomically adding the final contribution of the ray to the final image
                grad[0] += res_color[0] * collisions / differ_color[0]
                grad[1] += res_color[1] * collisions / differ_color[1]
                grad[2] += res_color[2] * collisions / differ_color[2]
                cuda.atomic.add(total_grad, 0, grad[0] * I_diff[j,i,0])
                cuda.atomic.add(total_grad, 1, grad[1] * I_diff[j,i,1])
                cuda.atomic.add(total_grad, 2, grad[2] * I_diff[j,i,2])


        @cuda.jit()
        def render_inverse_specular_cuda(object_mat, camera_location, width, height, spp, rng_states, I_diff, total_grad):
            i, j, s = cuda.grid(3)
            tid = height * width * s + width * j + i
            if i < width and j < height and s < spp:
                depth = 0
                res_color = cuda.local.array(3, dtype=np.float64)
                grad = cuda.local.array(2, dtype=np.float64)
                grad_temp = cuda.local.array(2, dtype=np.float64)
                current_point = cuda.local.array(3, dtype=np.float64)
                direction = cuda.local.array(3, dtype=np.float64)
                normal = cuda.local.array(3, dtype=np.float64)
                sampleDir = cuda.local.array(3, dtype=np.float64)
                specular_direction = cuda.local.array(3, dtype=np.float64)
                rotX = cuda.local.array(3, dtype=np.float64)
                rotY = cuda.local.array(3, dtype=np.float64)
                assign_value_3d(res_color, 1)
                grad[0] = 0
                grad[1] = 0
                grad_temp[0] = 0
                grad_temp[1] = 0
                assign_3d(current_point, camera_location)
                getPixelDirection(i, j, width, height, direction)
                calc_direction(camera_location, direction, direction)
                direct_flag = False
                obj_ind_temp = -1
                obj_n = -2.0
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
                    emission = object_mat[obj_ind, 8]
                    # if we have reached the light source, add the contribution and terminate
                    if emission > 0:
                        if not direct_flag:
                            multiply_by_scalar(res_color, emission * rrFactor)
                        break

                    # stepping towards the intersected object
                    step_in_direction(current_point, direction, mint)

                    # calc intesection normal
                    calc_normal(current_point, object_mat[obj_ind], normal)

                    # getting object properties
                    calc_normal(current_point, object_mat[obj_ind], normal)

                    obj_color = object_mat[obj_ind, 5:8]



                    obj_ind_temp = obj_ind
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
                                cosa = dot_3d(rotY, specular_direction)
                                brdf_specular = calc_specular_brdf(cosa, obj_n)
                                if diffusive:
                                    temp *= 2 * rad_squared * cost
                                else:
                                    temp *= brdf_specular * 2 * np.pi * rad_squared

                                # Grad calculation of Direct Illumination
                                if obj_ind == differ_ind:
                                    brdf = obj_Ks * brdf_specular + (1 - obj_Ks) * PI_INV

                                    if cosa > 0:
                                        # n gradient
                                        grad_direct = (obj_Ks * brdf_specular / brdf) * ( (1 / (obj_n + 1)) + math.log(cosa))
                                        grad_direct += grad_temp[0]
                                        grad[0] += res_color[0] * temp * grad_direct * I_diff[j,i,0]
                                        grad[0] += res_color[1] * temp * grad_direct * I_diff[j,i,1]
                                        grad[0] += res_color[2] * temp * grad_direct * I_diff[j,i,2]

                                        # Ks gradient
                                        grad_direct = (brdf_specular - PI_INV) / brdf
                                    else:
                                        grad_direct = -(1 / (1 - obj_Ks))
                                    # Ks gradient continued
                                    grad_direct += grad_temp[1]
                                    grad[1] += res_color[0] * temp * grad_direct * I_diff[j, i, 0]
                                    grad[1] += res_color[1] * temp * grad_direct * I_diff[j, i, 1]
                                    grad[1] += res_color[2] * temp * grad_direct * I_diff[j, i, 2]


                    # calc reflection gradient
                    if obj_ind == differ_ind:
                        cosa = dot_3d(direction, specular_direction)
                        brdf_specular = calc_specular_brdf(cosa, obj_n)
                        brdf = obj_Ks * brdf_specular + (1 - obj_Ks) * PI_INV
                        if cosa > 0:
                            grad_temp[0] += (obj_Ks * brdf_specular / brdf) * ((1 / (obj_n + 1)) + math.log(cosa))
                            grad_temp[1] += (brdf_specular - PI_INV) / brdf
                        else:
                            grad_temp[1] += -(1 / (1 - obj_Ks))

                    # increasing the ray depth
                    depth += 1

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

        self.render_cuda = render_cuda
        self.render_inverse_color_cuda = render_inverse_color_cuda
        self.render_inverse_specular_cuda = render_inverse_specular_cuda

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

    def render(self, spp, I_gt=None, inverse_type=None, init_rng=True):
        self.objects_to_device()
        I_res = np.zeros((self.camera.height, self.camera.width, 3), dtype=np.float64)
        dI_res = cuda.to_device(I_res)
        Np = int(I_res.shape[0]*I_res.shape[1]*spp)
        print(f"Np = {Np:.2e}")
        seed = np.random.randint(1, int(1e9))
        if init_rng:
            start = time()
            del(self.rng_states)
            self.rng_states = create_xoroshiro128p_states(Np, seed)
            print("init rng took:",time()-start)
        # threadsperblock = (8, 8, 8)
        threadsperblock = (6, 6, 6)
        blockspergrid_x = (self.camera.width + threadsperblock[0] -1) // threadsperblock[0]
        blockspergrid_y = (self.camera.height + threadsperblock[1] -1) // threadsperblock[1]
        blockspergrid_z = (spp + threadsperblock[2] -1) // threadsperblock[2]
        blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
        start = time()
        self.render_cuda[blockspergrid, threadsperblock](self.dobject_mat, self.dcamera_location, self.camera.width,
                                                         self.camera.height, spp, self.rng_states, dI_res)
        cuda.synchronize()
        print("render_cuda:", time()-start)
        I_res = dI_res.copy_to_host()/spp
        if I_gt is None:
            return I_res

        # differentiable part
        I_diff = I_res - I_gt
        dI_diff = cuda.to_device(I_diff)

        if inverse_type == "color":
            dtotal_grad = cuda.to_device(np.zeros(3, dtype=np.float64))
            inverse_func = self.render_inverse_color_cuda[blockspergrid, threadsperblock]
        elif inverse_type == "specular":
            dtotal_grad = cuda.to_device(np.zeros(2, dtype=np.float64))
            inverse_func = self.render_inverse_specular_cuda[blockspergrid, threadsperblock]
        else:
            print("inverse type is not supported")
            exit(1)
        start = time()
        inverse_func(self.dobject_mat, self.dcamera_location, self.camera.width, self.camera.height, spp, self.rng_states, dI_diff, dtotal_grad)

        total_grad = dtotal_grad.copy_to_host()/spp
        print("inverse_func took:", time() - start)
        return I_res,  total_grad



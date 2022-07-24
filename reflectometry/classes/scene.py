from utils import *
from classes.ray import Ray
import multiprocessing as mp
class Scene(object):
    def __init__(self, objects, camera):
        self.objects = objects
        self.camera = camera
        self.emitters = [obj for obj in self.objects if obj.emission>0]
        self.emitter_inds = [i for i in range(len(self.objects)) if self.objects[i].emission>0]
        self.non_emitter_objects = [obj for obj in self.objects if obj.emission==0]

    def intersect(self, ray:Ray):
        mint = np.inf
        obj_ind = -1
        for i, obj in enumerate(self.objects):
            t = obj.intersect(ray)
            if t > 0 and t < mint:
                mint = t
                obj_ind = i
        return mint, obj_ind

    def trace(self, ray, depth, color, random):
        rrFactor = 1.0
        # Russian Roulette
        if depth >= rr_depth:
            if random.rand(1) <= rr_stop_prob:
                return
            rrFactor = 1.0 / (1.0 - rr_stop_prob)

        mint, obj_ind = self.intersect(ray)
        hit_point = ray.origin + ray.direction * mint
        obj = self.objects[obj_ind]
        normal = obj.normal(hit_point)
        ray.origin = hit_point
        emission = obj.emission

        if emission > 0:
            color += np.ones(3) * emission * rrFactor
            return

        if obj.type == 1: # diffusive
            rotX , rotY = ons(normal)
            sampleDir = sample_hemisphere_uniform(random)
            rotatedDir = np.empty_like(sampleDir)
            rotatedDir[0] = np.dot(np.array([rotX[0], rotY[0], normal[0]]), sampleDir)
            rotatedDir[1] = np.dot(np.array([rotX[1], rotY[1], normal[1]]), sampleDir)
            rotatedDir[2] = np.dot(np.array([rotX[2], rotY[2], normal[2]]), sampleDir)
            ray.direction = rotatedDir
            # ray.direction = (normal + sampleDir)
            cost = np.dot(ray.direction, normal)
            if cost < 0:
                print("bug in cost:", cost)
            temp_color = np.zeros_like(color)
            self.trace(ray, depth+1, temp_color, random)
            temp_ray = ray.copy()
            color += temp_color * obj.color * cost * magic_constant * rrFactor
            # direct illumination
            rand_ind = 0 # for now
            direct_direction = self.emitters[rand_ind].center - temp_ray.origin
            direct_ray = Ray(temp_ray.origin, direct_direction)
            cost = np.dot(direct_ray.direction, normal)
            if cost > 0:
                mint, obj_ind = self.intersect(direct_ray)
                if mint > 0 and obj_ind in self.emitter_inds:
                    color += np.ones(3) * self.emitters[rand_ind].emission * cost * 0.0125 * rrFactor * (1/(2*np.pi))

        if obj.type == 2: # specular
            cost = np.dot(ray.direction, normal)
            new_direction = ray.direction - normal*(cost * 2)
            new_direction /= np.linalg.norm(new_direction)
            ray.direction = new_direction

            temp_color = np.zeros_like(color)
            self.trace(ray, depth + 1, temp_color, random)
            color += temp_color * rrFactor

        if obj.type == 3: #refractive
            pass

    def trace_nonRecursive(self, ray, random):
        depth = 0
        res_color = np.ones(3, dtype=np.float)
        while True:
            rrFactor = 1.0
            # Russian Roulette
            if depth >= rr_depth:
                if random.rand(1) <= rr_stop_prob:
                    return np.zeros(3, dtype=np.float)
                rrFactor = 1.0 / (1.0 - rr_stop_prob)

            mint, obj_ind = self.intersect(ray)
            hit_point = ray.origin + ray.direction * mint
            obj = self.objects[obj_ind]
            normal = obj.normal(hit_point)
            ray.origin = hit_point
            emission = obj.emission

            if emission > 0:
                res_color *= np.ones(3) * emission * rrFactor
                return res_color

            if obj.type == 1 or obj.type == 3: # diffusive
                rotX , rotY = ons(normal)
                if obj.type == 1:
                    sampleDir = sample_hemisphere_uniform(random)
                    is_specular = False
                else:
                    sampleDir, is_specular = sample_hemisphere_phong(random)
                rotatedDir = np.empty_like(sampleDir)
                rotatedDir[0] = np.dot(np.array([rotX[0], rotY[0], normal[0]]), sampleDir)
                rotatedDir[1] = np.dot(np.array([rotX[1], rotY[1], normal[1]]), sampleDir)
                rotatedDir[2] = np.dot(np.array([rotX[2], rotY[2], normal[2]]), sampleDir)
                ray.direction = rotatedDir
                # ray.direction = (normal + sampleDir)
                cost = np.dot(ray.direction, normal)
                if cost < 0:
                    print("bug in cost:", cost)
                # temp_ray = ray.copy()
                if not is_specular:
                    res_color *= obj.color * cost * magic_constant * rrFactor
                else:
                    res_color *= cost * rrFactor
                # direct illumination
                # rand_ind = 0 # for now
                # direct_direction = self.emitters[rand_ind].center - temp_ray.origin
                # direct_ray = Ray(temp_ray.origin, direct_direction)
                # cost = np.dot(direct_ray.direction, normal)
                # if cost > 0:
                #     mint, obj_ind = self.intersect(direct_ray)
                #     if mint > 0 and obj_ind in self.emitter_inds:
                #         color += np.ones(3) * self.emitters[rand_ind].emission * cost * 0.0125 * rrFactor * (1/(2*np.pi))

            if obj.type == 2: # specular
                cost = np.dot(ray.direction, normal)
                new_direction = ray.direction - normal*(cost * 2)
                new_direction /= np.linalg.norm(new_direction)
                ray.direction = new_direction

                res_color *= rrFactor
            depth += 1




    def render(self, spp, workers=1):
        I_res = np.zeros((self.camera.height, self.camera.width, 3), dtype=np.float)
        if workers == 1:
            for s in tqdm(range(spp)):
                for i in range(self.camera.width):
                    for j in range(self.camera.height):
                        origin = np.array([0.0,0.0,0.0])
                        pixel_vec = self.camera.getPixelDirection(i, j)
                        direction = pixel_vec - origin
                        direction /= np.linalg.norm(direction)
                        ray = Ray(origin, direction)
                        color = np.zeros(3, dtype=np.float)
                        # self.trace(ray, 0, color)
                        self.trace_nonRecursive(ray)
                        I_res[j,i,:] += color
        else:
            pix_num = self.camera.width*self.camera.height
            params = [(tqdm_wrapper,((self, pixel_ind, spp),)) for pixel_ind in range(pix_num)]
            pool = TqdmMultiProcessPool(workers)
            with tqdm(total=pix_num, dynamic_ncols=True) as global_prgoress:
                # results = pool.map(global_prgoress, params, error_callback, done_callback)
                # pass
                for pixel_ind, color in enumerate(pool.map(global_prgoress, params, error_callback, done_callback)):
                    i = pixel_ind % self.camera.width
                    j = pixel_ind // self.camera.width
                    I_res[j,i,:] += color[0]
        I_res /= spp
        return I_res


def tqdm_wrapper(args, tqdm_func, global_tqdm):
    res = render_loop_mp(args)
    global_tqdm.update()
    return (res,)

def render_loop_mp(args):
    random = np.random.RandomState()
    self, pixel_ind, spp = args
    i = pixel_ind % self.camera.width
    j = pixel_ind // self.camera.width
    origin = np.array([0.0, 0.0, 0.0])
    pixel_vec = self.camera.getPixelDirection(i, j)
    res = np.zeros(3, dtype=np.float)
    for s in range(spp):
        rands = (random.rand(2) - 0.5)*2
        pixel_vec_temp = np.copy(pixel_vec)
        pixel_vec_temp[0] += rands[0]/700
        pixel_vec_temp[1] += rands[1]/700
        direction = pixel_vec_temp - origin
        direction /= np.linalg.norm(direction)
        ray = Ray(origin, direction)

        # color = np.zeros(3, dtype=np.float)
        # self.trace(ray.copy(), 0, color, random)
        color = self.trace_nonRecursive(ray, random)
        res += color
    return res


def error_callback(result):
    print("error")

def done_callback(result):
    pass
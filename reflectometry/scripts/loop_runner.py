import sys
from os.path import join
import os


# scene = "specular"
scene = "albedo"
# renderer = "rr"
renderer = "seed"
script_name = f"{renderer}_inverse_{scene}_loop.py"
script_path = join("scripts", script_name)
# Nrs =      [10]
Nrs =      [10]
# Nrs =      [10, 1, 2, 5, 20 ,30]
# Nrs =      [1, 2, 5, 20 ,30]
# Nrs =      [5, 10, 20 ,30]
runtimes = [120] * len(Nrs)
print("Nrs=",Nrs)
print("rumtimes=",runtimes)
for to_sort in [1]:
    for Nr, runtime in zip(Nrs, runtimes):
        print("###############################")
        print("###############################")
        print(f"############Nr={Nr}#################")
        print("###############################")
        print("###############################")
        # if Nr == 1:
        #     to_sort = 0
        # else:
        #     to_sort = 1
        print("to_sort=", to_sort)
        try:
            script_command = f"python {script_path} {Nr} {runtime} {to_sort}"
            print("running:", script_command)
            os.system(script_command)
        except KeyboardInterrupt:
            continue
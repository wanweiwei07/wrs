from wrs import wd, rm, mcm
import wrs.robot_sim.robots.cobotta.cobotta as cbt
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

base = wd.World(cam_pos=[1.7, 1.7, 1.7], lookat_pos=[0, 0, .3])
mcm.mgm.gen_frame().attach_to(base)

robot = cbt.Cobotta(pos=rm.vec(0.1,.3,.5), enable_cc=True)
nupdate = 100

while True:
    success_rate = 0
    time_list = []
    for i in tqdm(range(nupdate)):
        jnt_values = robot.rand_conf()
        tgt_pos, tgt_rotmat = robot.fk(jnt_values = jnt_values)
        tic = time.time()
        result = robot.ik(tgt_pos, tgt_rotmat)
        toc = time.time()
        time_list.append(toc-tic)
        if result is not None:
            success_rate += 1

    print(success_rate)
    plt.plot(range(nupdate), time_list)
    plt.show()
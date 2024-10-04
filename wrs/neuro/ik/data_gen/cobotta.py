import math
import itertools
import numpy as np
from wrs import basis as rm, robot_sim as cbt_s, modeling as gm
import wrs.visualization.panda.world as world


# file size: pandas (string) > pickle (binary) = torch.save > numpy, 20211216

def gen_data(rbt_s, component_name='arm', granularity=math.pi / 8, save_name='cobotta_ik.csv'):
    n_jnts = rbt_s.manipulator_dict[component_name].n_dof
    all_ranges = []
    for jnt_id in range(1, n_jnts + 1):
        r0, r1 = rbt_s.manipulator_dict[component_name].jnts[jnt_id]['motion_range']
        all_ranges.append(np.arange(r0, r1, granularity))
        # print(granularity, all_ranges[-1])
    all_data = itertools.product(*all_ranges)
    n_data = 1
    for rngs in all_ranges:
        n_data = n_data * len(rngs)
    data_set = []
    in_data_npy = np.empty((0, 6))
    for i, data in enumerate(all_data):
        print(i, n_data)
        rbt_s.fk(component_name=component_name, joint_values=np.array(data))
        xyz, rotmat = rbt_s.get_gl_tcp(manipulator_name=component_name)
        rpy = rm.rotmat_to_euler(rotmat)
        in_data = (xyz[0], xyz[1], xyz[2], rpy[0], rpy[1], rpy[2])
        # diff = np.sum(np.abs(np.array(in_data) - in_data_npy), 1)
        # if np.any(diff < 1e-4):
        #     print(diff)
        #     input("Press Enter to continue...")
        in_data_npy = np.vstack((in_data_npy, np.array(in_data)))
        out_data = data
        data_set.append([in_data, out_data])
    # df = pd.DataFrame(data_set, columns=['xyzrpy', 'jnt_values'])
    # df.to_csv(save_name)
    np.save(save_name+"_min_max", np.array([np.min(in_data_npy, 0), np.max(in_data_npy, 0)]))
    np.save(save_name, np.array(data_set))


if __name__ == '__main__':
    base = world.World(cam_pos=np.array([1.5, 1, .7]))
    gm.gen_frame().attach_to(base)
    rbt_s = cbt_s.Cobotta()
    rbt_s.gen_meshmodel(toggle_tcp_frame=True).attach_to(base)
    gen_data(rbt_s, granularity=math.pi / 4, save_name='cobotta_ik')
    gen_data(rbt_s, granularity=math.pi / 4, save_name='cobotta_ik_test')
    base.run()

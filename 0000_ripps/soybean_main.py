import copy
import math
import numpy as np
import random
from wrs import basis as rm, robot_sim as jlc, robot_sim as rbt, modeling as gm, modeling as cm
import wrs.visualization.panda.world as wd

leaf_rgba = [45 / 255, 90 / 255, 39 / 255, 1]
stem_rgba = [97 / 255, 138 / 255, 61 / 255, 1]

aluminium_rgba = [132 / 255, 135 / 255, 137 / 255, 1]
board_rgba = [235 / 255, 235 / 255, 205 / 255, 1]
earth_rgba = [66 / 255, 40 / 255, 14 / 255, 1]

matt_red = [162 / 255, 32 / 255, 65 / 255, 1]
matt_blue = [30 / 255, 80 / 255, 162 / 255, 1]
matt_black = [44 / 255, 44 / 255, 44 / 255, 1]
reflective_black = [74 / 255, 74 / 255, 74 / 255, 1]

matt_purple = [204 / 255, 126 / 255, 177 / 255, 1]
red = [236 / 255, 104 / 255, 28 / 255, 1]

class Cup(object):

    def __init__(self):
        self.cup = cm.CollisionModel("objects/cup.stl")
        self.earth = cm.CollisionModel("objects/earth.stl")
        self.earth.set_rgba(rgba=earth_rgba)

    def set_pos(self, pos):
        self.cup.set_pos(pos=pos)
        self.earth.set_pos(pos=pos)

    def attach_to(self, base):
        self.cup.attach_to(base)
        self.earth.attach_to(base)

    def copy(self):
        return copy.deepcopy(self)


base = wd.World(cam_pos=[4.2, 4.2, 2.5], lookat_pos=[0, 0, .7], auto_cam_rotate=True)
frame = gm.GeometricModel(initor="meshes/frame.stl")
frame.set_rgba(rgba=aluminium_rgba)
frame.attach_to(base)

bottom_box = cm.gen_box(xyz_lengths=[.88, 1.68, .45], rgba=board_rgba)
bottom_box.set_pos(pos=np.array([0, 0, .22]))
bottom_box.attach_to(base)

top_box = cm.gen_box(xyz_lengths=[.88, 1.68, .3], rgba=board_rgba)
top_box.set_pos(pos=np.array([0, 0, 1.65]))
top_box.attach_to(base)

cup = Cup()
cup_pos_x = [0.09 - .44, 0.23 - .44, 0.37 - .44, 0.51 - .44, 0.65 - .44, 0.79 - .44]
cup_pos_y = [.09 - .84, .24 - .84, .39 - .84, .54 - .84, .69 - .84, .84 - .84, .99 - .84, 1.14 - .84, 1.29 - .84,
             1.44 - .84, 1.59 - .84]
# cup_pos_x = [ 0.23 - .44, 0.51 - .44, 0.79 - .44]
# cup_pos_y = [.24 - .84, .54 - .84, .84 - .84, 1.14 - .84,
#              1.44 - .84]
cup_pos_z = .37
for idx, x in enumerate(cup_pos_x):
    for idy, y in enumerate(cup_pos_y):
        pos = np.array([x, y, cup_pos_z])
        current_cup = cup.copy()
        current_cup.set_pos(pos=pos)
        current_cup.attach_to(base)
        if idy % 2 == 1 or idx % 2 == 0:
            current_cup.earth.set_rgba([1, 1, 1, 1])


# soybean plant
class Stem(object):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), ndof=5, base_thickness=.005, base_length=.3, name='stem'):
        self.jlc = jlc.JLChain(pos=pos, rotmat=rotmat, home_conf=np.zeros(ndof), name=name + "jlchain")
        for i in range(1, self.jlc.n_dof + 1):
            self.jlc.jnts[i]['loc_pos'] = np.array([0, 0, base_length / 5])
            self.jlc.jnts[i]['loc_motionax'] = np.array([1, 0, 0])
        self.jlc.finalize()
        for link_id in range(self.jlc.n_dof + 1):
            self.jlc.lnks[link_id]['collision_model'] = cm.gen_stick(spos=np.zeros(3),
                                                                     epos=rotmat.T.dot(
                                                                         self.jlc.jnts[link_id + 1]['gl_posq'] -
                                                                         self.jlc.jnts[link_id]['gl_posq']),
                                                                     radius=base_thickness / (link_id + 1) ** (
                                                                             1 / 3),
                                                                     n_sec=24)

    def fk(self, jnt_values):
        self.jlc.fk(joint_values=jnt_values)

    def gen_meshmodel(self,
                      toggle_tcp_frame=False,
                      toggle_jnt_frame=False,
                      rgba=stem_rgba,
                      name='stem_meshmodel'):
        return self.jlc.gen_mesh_model(toggle_tcp_frame=toggle_tcp_frame, toggle_jnt_frame=toggle_jnt_frame, name=name, rgba=rgba)

    def gen_stickmodel(self):
        return self.jlc.gen_stickmodel()

    def copy(self):
        self_copy = copy.deepcopy(self)
        return self_copy


def gen_rotmat_list(nsample=None):
    rotmats = rm.gen_icorotmats(icolevel=2,
                                rotation_interval=math.radians(30),
                                crop_normal=np.array([0, 0, -1]),
                                crop_angle=np.pi / 3,
                                toggle_flat=True)
    return_rotmat = []
    for rotmat in rotmats:
        if rm.angle_between_vectors(rotmat[:, 0], np.array([0, 0, -1])) < np.pi / 3:
            return_rotmat.append(rotmat)
    nreturn = len(return_rotmat)
    if nsample is not None and nsample < nreturn:
        return return_rotmat[0:nreturn:int(nreturn / nsample)]
    return return_rotmat


rbt_s = rbt.XArm7(pos=np.array([.42, -0.4, 1.5]), rotmat=rm.rotmat_from_axangle([0, 1, 0], np.pi))
# goal_pos = np.array([.7, -.5, 1.2])
# goal_rotmat = rm.rotmat_from_axangle([0, 1, 0], np.pi * 7 / 6)
goal_pos = np.array([.1, -.4, 1.1])
goal_rotmat = rm.rotmat_from_axangle([0, 1, 0], np.pi * 5 / 6)
# goal_pos = np.array([.6, -.5, 1.1])
# goal_rotmat = rm.rotmat_from_axangle([0, 1, 0], np.pi * 7 / 6)
# goal_rotmat = rm.rotmat_from_axangle(goal_rotmat[:,2], math.pi/2).dot(goal_rotmat)
# mgm.gen_frame(pos=goal_pos, rotmat=goal_rotmat).attach_to(base)
jnt_values = rbt_s.ik(tgt_pos=goal_pos, tgt_rotmat=goal_rotmat, max_niter=500)
if jnt_values is not None:
    rbt_s.fk(jnt_values=jnt_values)
else:
    base.run()
# mgm.gen_frame(pos=goal_pos, rotmat = goal_rotmat).attach_to(base)
rbt_model = rbt_s.gen_meshmodel(toggle_tcp_frame=False)
rbt_model.attach_to(base)
# base.run()

pos, rotmat = rbt_s.get_gl_tcp()
r_rotmat = rotmat.dot(rm.rotmat_from_axangle([0, 1, 0], -np.pi / 2))
# pos = np.zeros(3)
# rotmat = np.eye(3)

cam_frame = cm.CollisionModel(initor="objects/camera_frame.stl")
cam_frame.set_rgba(rgba=aluminium_rgba)
cam_frame.set_pose(pos, r_rotmat)
cam_frame.attach_to(base)

cam_0 = cm.CollisionModel(initor="objects/flircam.stl")
cam_1 = cam_0.copy()
cam_2 = cam_0.copy()
phoxi = cm.CollisionModel(initor="objects/phoxi_m.stl")

cam_0.set_rgba(rgba=matt_red)
cam_1.set_rgba(rgba=matt_blue)
cam_2.set_rgba(rgba=matt_black)
phoxi.set_rgba(rgba=reflective_black)

phoxi_loc_pos = np.array([.12, 0, 0.015])
phoxi_loc_rotmat = rm.rotmat_from_axangle(r_rotmat[:, 0], np.pi)
phoxi_gl_rotmat = phoxi_loc_rotmat.dot(r_rotmat)
phoxi_gl_pos = phoxi_gl_rotmat.dot(phoxi_loc_pos) + pos
phoxi.set_pose(pos=phoxi_gl_pos, rotmat=phoxi_gl_rotmat)
phoxi.attach_to(base)

cam_0_loc_pos = np.array([.075, .075, .03])
cam_0_loc_rotmat = np.eye(3)
cam_1_loc_pos = np.array([.075, -.075, .03])
cam_1_loc_rotmat = np.eye(3)
cam_2_loc_pos = np.array([.095, 0, .03])
cam_2_loc_rotmat = np.eye(3)

cam_0_gl_rotmat = cam_0_loc_rotmat.dot(r_rotmat)
cam_0_gl_pos = cam_0_gl_rotmat.dot(cam_0_loc_pos) + pos
cam_1_gl_rotmat = cam_1_loc_rotmat.dot(r_rotmat)
cam_1_gl_pos = cam_1_gl_rotmat.dot(cam_1_loc_pos) + pos
cam_2_gl_rotmat = cam_2_loc_rotmat.dot(r_rotmat)
cam_2_gl_pos = cam_2_gl_rotmat.dot(cam_2_loc_pos) + pos
cam_0.set_pose(pos=cam_0_gl_pos, rotmat=cam_0_gl_rotmat)
cam_1.set_pose(pos=cam_1_gl_pos, rotmat=cam_1_gl_rotmat)
cam_2.set_pose(pos=cam_2_gl_pos, rotmat=cam_2_gl_rotmat)

cam_0.attach_to(base)
cam_1.attach_to(base)
cam_2.attach_to(base)
cam_0.attach_to(base)
cam_1.attach_to(base)

rbt_s = rbt.XArm7(pos=np.array([.42, .3, 1.5]), rotmat=rm.rotmat_from_axangle([0, 1, 0], np.pi))
goal_pos = np.array([.8, .4, .8])
goal_rotmat = rm.rotmat_from_axangle([0, 1, 0], np.pi * 8 / 6)
goal_rotmat = rm.rotmat_from_axangle([0,0,1], np.pi/6).dot(rm.rotmat_from_axangle(goal_rotmat[:,2], -np.pi/2).dot(goal_rotmat))
jnt_values = rbt_s.ik(tgt_pos=goal_pos, tgt_rotmat=goal_rotmat, max_niter=500)
if jnt_values is not None:
    rbt_s.fk(jnt_values=jnt_values)
else:
    base.run()
rbt_model = rbt_s.gen_meshmodel(toggle_tcp_frame=False)
rbt_model.attach_to(base)

pos, rotmat = rbt_s.get_gl_tcp()
r_rotmat = rotmat

spray_host = cm.CollisionModel(initor="objects/airgun_host.stl")
spray_host.set_rgba(rgba=[1,1,1,1])
spray_host.set_pose(pos, r_rotmat)
spray_host.attach_to(base)

spray = cm.CollisionModel(initor="objects/spray.stl")
spray_loc_pos = np.array([.0, -.07, 0.07])
spray_loc_rotmat = np.eye(3)
spray_gl_rotmat = spray_loc_rotmat.dot(r_rotmat)
spray_gl_pos = spray_gl_rotmat.dot(spray_loc_pos) + pos
spray.set_pose(pos=spray_gl_pos, rotmat=spray_gl_rotmat)
spray.attach_to(base)
spray.set_rgba(red)

container = cm.CollisionModel(initor="objects/spray_container.stl")
container_loc_pos = np.array([.0, -.01, 0.15])
container_loc_rotmat = np.eye(3)
container_gl_rotmat = container_loc_rotmat.dot(r_rotmat)
container_gl_pos = container_gl_rotmat.dot(container_loc_pos) + pos
container.set_pose(pos=container_gl_pos, rotmat=container_gl_rotmat)
container.attach_to(base)
container.set_rgba(matt_purple)
# goal_pos = np.array([.7, .3, 1.2])
# goal_rotmat = rm.rotmat_from_axangle([0, 1, 0], np.pi * 7 / 6)

for idx, x in enumerate(cup_pos_x[1::2]):
    for idy, y in enumerate(cup_pos_y[::2]):
        #         # cup_p = np.array([cup_pos_x[-1], cup_pos_y[5], cup_pos_z+.1])
        #         idx = 5
        #         idy = 5
        #         x = cup_pos_x[idx]
        #         y = cup_pos_y[idy]
        id_all = idx * 11 + idy
        print(id_all)
        main_stem_ndof = 5
        cup_p = np.array([x, y, cup_pos_z + .1])
        main_stem = Stem(pos=cup_p, rotmat=rm.rotmat_from_axangle([0, 0, 1], np.pi * ((id_all + 1) / 30)),
                         ndof=main_stem_ndof)
        main_stem.fk(jnt_values=[math.pi / 36, math.pi / 36, 0, -math.pi / 36, -math.pi / 36, 0])
        main_stem.gen_meshmodel().attach_to(base)
        rotmat_list = gen_rotmat_list(2 ** main_stem_ndof)
        for id, rotmat in enumerate(random.sample(rotmat_list, len(rotmat_list))):
            branch_pos = main_stem.jlc.jnts[int(id / main_stem_ndof) % (main_stem.jlc.n_dof + 1) + 1]['gl_posq']
            height = branch_pos[2] - main_stem.jlc.pos[2]
            branch = Stem(ndof=1, pos=branch_pos,
                          rotmat=rotmat, base_length=.1 / math.sqrt(height), base_thickness=.002)
            branch.gen_meshmodel().attach_to(base)
            sb_leaf = gm.GeometricModel(initor="objects/soybean_leaf.stl")
            sb_leaf.set_rgba(rgba=leaf_rgba)
            sbl = sb_leaf.copy()
            # sbl.set_scale(np.array([1,1,1])/(int(id/3)%(main_stem.jlc.n_dof+1)+1))
            sbl.set_scale(np.array([1, 1, 1]))
            jnt_pos = branch.jlc.jnts[-1]['gl_posq']
            sbl.set_pos(jnt_pos)
            sbl.set_rotmat(rotmat)
            sbl.attach_to(base)

base.run()

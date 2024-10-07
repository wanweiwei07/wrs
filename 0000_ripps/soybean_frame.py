import copy
import math
import numpy as np
from wrs import basis as rm, robot_sim as jlc, modeling as gm, modeling as cm
import wrs.visualization.panda.world as wd

leaf_rgba = [45 / 255, 90 / 255, 39 / 255, 1]
stem_rgba = [97 / 255, 138 / 255, 61 / 255, 1]

aluminium_rgba = [132 / 255, 135 / 255, 137 / 255, 1]
board_rgba = [235 / 255, 235 / 255, 205 / 255, 1]
earth_rgba = [66 / 255, 40 / 255, 14 / 255, 1]


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


base = wd.World(cam_pos=[7, 7, 2], auto_cam_rotate=False)
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
for x in cup_pos_x:
    for y in cup_pos_y:
        pos = np.array([x, y, cup_pos_z])
        current_cup = cup.copy()
        current_cup.set_pos(pos=pos)
        current_cup.attach_to(base)


# soybean plant
class Stem(object):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), ndof=5, base_thickness=.005, base_length=.3, name='stem'):
        self.jlc = jlc.JLChain(pos=pos, rotmat=rotmat, home_conf=np.zeros(ndof), name=name + "jlchain")
        for i in range(1, self.jlc.n_dof + 1):
            self.jlc.jnts[i]['loc_pos'] = np.array([0, 0, base_length / ndof])
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


for idx, x in enumerate(cup_pos_x[1::2]):
    for idy, y in enumerate(cup_pos_y[::2]):
        # cup_p = np.array([cup_pos_x[-1], cup_pos_y[5], cup_pos_z+.1])
        id_all = idx * 11 + idy
        print(id_all)
        main_stem_ndof = 5
        cup_p = np.array([x, y, cup_pos_z + .1])
        main_stem = Stem(pos=cup_p, rotmat=rm.rotmat_from_axangle([0,0,1], math.pi/36*id_all), ndof=main_stem_ndof)
        main_stem.fk(jnt_values=[math.pi / 36, math.pi / 36, 0, -math.pi / 36, -math.pi / 36, 0])
        main_stem.gen_meshmodel().attach_to(base)
        rotmat_list = gen_rotmat_list(2 ** main_stem_ndof)

        for id, rotmat in enumerate(rotmat_list):
            stem1 = Stem(ndof=1, pos=main_stem.jlc.jnts[int(id / 3) % (main_stem.jlc.n_dof + 1) + 1]['gl_posq'],
                         rotmat=rotmat, base_length=.2 / (id + 1) ** (1 / 2), base_thickness=.002)
            stem1.gen_meshmodel().attach_to(base)
            sb_leaf = gm.GeometricModel(initor="objects/soybean_leaf.stl")
            sb_leaf.set_rgba(rgba=leaf_rgba)
            sbl = sb_leaf.copy()
            # sbl.set_scale(np.array([1,1,1])/(int(id/3)%(main_stem.jlc.n_dof+1)+1))
            sbl.set_scale(np.array([1, 1, 1]))
            jnt_pos = stem1.jlc.jnts[-1]['gl_posq']
            sbl.set_pos(jnt_pos)
            sbl.set_rotmat(rotmat)
            sbl.attach_to(base)

base.run()

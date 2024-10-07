import copy
import math
import numpy as np
import random
from wrs import basis as rm, robot_sim as jlc, modeling as gm, modeling as cm
import wrs.visualization.panda.world as wd

leaf_rgba = [45 / 255, 90 / 255, 39 / 255, 1]
stem_rgba = [97 / 255, 138 / 255, 61 / 255, 1]
earth_rgba = [66 / 255, 40 / 255, 14 / 255, 1]
wither_rgba = [145/255, 115/255, 71/255, 1]


class Stem(object):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), ndof=5, base_thickness=.005, base_length=.3, name='stem'):
        self.pos = pos
        self.rotmat = rotmat
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

def gen_wither_rotmat_list(nsample=None):
    rotmats = rm.gen_icorotmats(icolevel=3,
                                rotation_interval=math.radians(3),
                                crop_normal=np.array([0, 0, 1]),
                                crop_angle=np.pi*1.3/3,
                                toggle_flat=True)
    # return rotmats
    return_rotmat = []
    for rotmat in rotmats:
        if rm.angle_between_vectors(rotmat[:, 0], np.array([0, 0, -1])) < np.pi / 10:
            return_rotmat.append(rotmat)
    nreturn = len(return_rotmat)
    if nsample is not None and nsample < nreturn:
        return return_rotmat[0:nreturn:int(nreturn / nsample)]
    return return_rotmat

map_list = [leaf_rgba]
#
# water_blue1 = [65 / 255, 107 / 255, 140 / 255, 1]
# water_blue2 = [65 / 255, 127 / 255, 160 / 255, 1]
# water_blue3 = [65 / 255, 147 / 255, 180 / 255, 1]
# water_blue4 = [65 / 255, 167 / 255, 200 / 255, 1]
# water_blue5 = [65 / 255, 187 / 255, 220 / 255, 1]
# map_list = [water_blue1, water_blue2, water_blue3, water_blue4, water_blue5]
# #
# water_blue1 = [240 / 255, 200 / 255, 20 / 255, 1]
# water_blue2 = [240 / 255, 160 / 255, 20 / 255, 1]
# water_blue3 = [240 / 255, 120 / 255, 20 / 255, 1]
# water_blue4 = [240 / 255, 80 / 255, 20 / 255, 1]
# water_blue5 = [240 / 255, 40 / 255, 20 / 255, 1]
# map_list = [water_blue5, water_blue4, water_blue3, water_blue2, water_blue1]
# # #
# water_blue1 = [107 / 255, 65 / 255, 140 / 255, 1]
# water_blue2 = [127 / 255, 65 / 255, 160 / 255, 1]
# water_blue3 = [147 / 255, 65 / 255, 180 / 255, 1]
# water_blue4 = [167 / 255, 65 / 255, 200 / 255, 1]
# water_blue5 = [187 / 255, 65 / 255, 220 / 255, 1]
# map_list = [water_blue1, water_blue2, water_blue3, water_blue4, water_blue5]

main_stem_ndof = 4

base = wd.World(cam_pos=[1, 1, 1], auto_cam_rotate=False)

# cup=Cup()
# cup.attach_to(base)
# base.run()

main_stem = Stem(pos=np.array([0, 0, .1]), ndof=main_stem_ndof)
main_stem.fk(jnt_values=[math.pi / 36, math.pi / 36, 0, -math.pi / 36, -math.pi / 36, 0, 0,0,0,0,0])
main_stem.gen_meshmodel(rgba=wither_rgba).attach_to(base)

rotmat_list = gen_rotmat_list(2 ** main_stem_ndof)
wither_rotmat_list = gen_wither_rotmat_list(2 ** main_stem_ndof)
print(len(wither_rotmat_list))

for id, rotmat in enumerate(rotmat_list):
    map_color = random.choice(map_list)
    # print(int(id +1) % 4)
    # print(int(id / 3+1) % (main_stem.jlc.n_dof + 1))
    # stem1 = Stem(n_dof=1, pos=main_stem.jlc.joints[int(id / 3) % (main_stem.jlc.n_dof + 1)+1]['gl_posq'], rotmat=rotmat, base_length=.2/ (id + 1) ** (1 / 2), base_thickness=.002)
    branch_pos = main_stem.jlc.jnts[int(id / main_stem_ndof) % (main_stem.jlc.n_dof + 1) + 1]['gl_posq']
    height = branch_pos[2] - main_stem.jlc.pos[2]
    print(height)
    # 4,5
    # if height > .05:
    #     select_id = 0
    # if height > .1:
    #     select_id = 1
    # if height > .15:
    #     select_id = 2
    # if height > .2:
    #     select_id = 3
    # if height > .25:
    #     select_id = 4
    # 3
    # if height > .03:
    #     select_id = 0
    # if height > .06:
    #     select_id = 1
    # if height > .09:
    #     select_id = 2
    # if height > .12:
    #     select_id = 3
    # if height > .15:
    #     select_id = 4
    # 1, 2
    # if height > .02:
    #     select_id = 0
    # if height > .04:
    #     select_id = 1
    # if height > .06:
    #     select_id = 2
    # if height > .08:
    #     select_id = 3
    # if height > .1:
    #     select_id = 4
    # map_color = map_list[select_id]
    if height < .25:
        this_rotmat = wither_rotmat_list[id%len(wither_rotmat_list)]
        branch = Stem(ndof=1, pos=branch_pos,
                      rotmat=this_rotmat, base_length=.1 / (height)**(1/2), base_thickness=.002)
        branch.gen_meshmodel(rgba=wither_rgba).attach_to(base)
        # main_stem.fk(jnt_values=[math.pi/36,math.pi/36, 0,-math.pi/36,-math.pi/36,0])
        # stem1.gen_meshmodel().attach_to(base)

        sb_leaf = gm.GeometricModel(initor="objects/soybean_leaf.stl")
        sb_leaf.set_rgba(rgba=leaf_rgba)
        sbl = sb_leaf.copy()
        sbl.set_rgba(rgba=wither_rgba)
        # sbl.set_scale(np.array([1,1,1])/(int(id/3)%(main_stem.jlc.n_dof+1)+1))
        sbl.set_scale(np.array([1, 1, 1]))
        jnt_pos = branch.jlc.jnts[-1]['gl_posq']
        sbl.set_pos(jnt_pos)
        sbl.set_rotmat(this_rotmat)
        # mgm.gen_frame(pos=jnt_pos, rotmat=this_rotmat).attach_to(base)
        sbl.attach_to(base)
    else:
        branch = Stem(ndof=1, pos=branch_pos,
                      rotmat=rotmat, base_length=.1 / (height)**(1/2), base_thickness=.002)
        branch.gen_meshmodel(rgba=map_color).attach_to(base)
        # main_stem.fk(jnt_values=[math.pi/36,math.pi/36, 0,-math.pi/36,-math.pi/36,0])
        # stem1.gen_meshmodel().attach_to(base)

        sb_leaf = gm.GeometricModel(initor="objects/soybean_leaf.stl")
        sb_leaf.set_rgba(rgba=leaf_rgba)
        sbl = sb_leaf.copy()
        sbl.set_rgba(rgba=map_color)
        # sbl.set_scale(np.array([1,1,1])/(int(id/3)%(main_stem.jlc.n_dof+1)+1))
        sbl.set_scale(np.array([1, 1, 1]))
        jnt_pos = branch.jlc.jnts[-1]['gl_posq']
        sbl.set_pos(jnt_pos)
        sbl.set_rotmat(rotmat)
        # mgm.gen_frame(pos=jnt_pos, rotmat=rotmat).attach_to(base)
        sbl.attach_to(base)

sbl_cup = Cup()
sbl_cup.attach_to(base)

base.run()

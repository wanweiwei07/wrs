import copy
import math
import numpy as np
from wrs import basis as rm, robot_sim as jlc, modeling as gm, modeling as cm
import wrs.visualization.panda.world as wd

leaf_rgba = [45 / 255, 90 / 255, 39 / 255, 1]
stem_rgba = [97 / 255, 138 / 255, 61 / 255, 1]


class Stem(object):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), ndof=5, base_thickness=.005, base_length=.3, name='stem'):
        self.pos = pos
        self.rotmat = rotmat
        self.jlc = jlc.JLChain(pos=pos, rotmat=rotmat, home_conf=np.zeros(ndof), name=name + "jlchain")
        for i in range(1, self.jlc.n_dof + 1):
            self.jlc.jnts[i]['loc_pos']=np.array([0, 0, base_length / 5])
            self.jlc.jnts[i]['loc_motionax']=np.array([1, 0, 0])
        self.jlc.finalize()
        for link_id in range(self.jlc.n_dof + 1):
            self.jlc.lnks[link_id]['collision_model'] = cm.gen_stick(spos=np.zeros(3),
                                                                     epos=rotmat.T.dot(self.jlc.jnts[link_id + 1]['gl_posq'] - self.jlc.jnts[link_id]['gl_posq']),
                                                                     radius=base_thickness / (link_id + 1) ** (1 / 3),
                                                                     n_sec=24)

    def fk(self, jnt_values):
        self.jlc.fk(joint_values=jnt_values)

    def gen_meshmodel(self,
                      toggle_tcpcs=False,
                      toggle_jntscs=False,
                      rgba=stem_rgba,
                      name='stem_meshmodel'):
        return self.jlc.gen_mesh_model(toggle_tcpcs=toggle_tcpcs, toggle_jntscs=toggle_jntscs, name=name, rgba=rgba)

    def gen_stickmodel(self):
        return self.jlc.gen_stickmodel()

    def copy(self):
        self_copy = copy.deepcopy(self)
        return self_copy

def gen_rotmat_list(nsample=None):
    rotmats = rm.gen_icorotmats(icolevel=2,
                                rotation_interval=math.radians(30),
                                crop_normal=np.array([0,0,-1]),
                                crop_angle=np.pi / 3,
                                toggle_flat=True)
    print(len(rotmats))
    return_rotmat = []
    for rotmat in rotmats:
        if rm.angle_between_vectors(rotmat[:,0], np.array([0,0,-1]))<np.pi/3:
            return_rotmat.append(rotmat)
    nreturn = len(return_rotmat)
    print(len(return_rotmat))
    if nsample is not None and nsample<nreturn:
        return return_rotmat[0:nreturn:int(nreturn/nsample)]
    return return_rotmat

main_stem_ndof = 5

base = wd.World(cam_pos=[1, 1, 1], auto_cam_rotate=True)
main_stem = Stem(ndof=main_stem_ndof)
main_stem.fk(jnt_values=[math.pi / 36, math.pi / 36, 0, -math.pi / 36, -math.pi / 36, 0])
main_stem.gen_meshmodel().attach_to(base)

rotmat_list =gen_rotmat_list(2**main_stem_ndof)

for id, rotmat in enumerate(rotmat_list):
    # print(int(id +1) % 4)
    # print(int(id / 3+1) % (main_stem.jlc.n_dof + 1))
    # stem1 = Stem(n_dof=1, pos=main_stem.jlc.joints[int(id / 3) % (main_stem.jlc.n_dof + 1)+1]['gl_posq'], rotmat=rotmat, base_length=.2/ (id + 1) ** (1 / 2), base_thickness=.002)
    branch_pos = main_stem.jlc.jnts[int(id / main_stem_ndof) % (main_stem.jlc.n_dof + 1) + 1]['gl_posq']
    height = branch_pos[2]-main_stem.jlc.pos[2]
    print(height)
    branch = Stem(ndof=1, pos=branch_pos,
                  rotmat=rotmat, base_length=.1 / math.sqrt(height), base_thickness=.002)
    branch.gen_meshmodel().attach_to(base)
    # main_stem.fk(jnt_values=[math.pi/36,math.pi/36, 0,-math.pi/36,-math.pi/36,0])
    # stem1.gen_meshmodel().attach_to(base)

    sb_leaf = gm.GeometricModel(initor="objects/soybean_leaf.stl")
    sb_leaf.set_rgba(rgba=leaf_rgba)
    sbl = sb_leaf.copy()
    # sbl.set_scale(np.array([1,1,1])/(int(id/3)%(main_stem.jlc.n_dof+1)+1))
    sbl.set_scale(np.array([1,1,1]))
    jnt_pos = branch.jlc.jnts[-1]['gl_posq']
    sbl.set_pos(jnt_pos)
    sbl.set_rotmat(rotmat)
    sbl.attach_to(base)

# for jnt_id in range(1, stem1.jlc.n_dof+2):
#     sbl = sb_leaf.copy()
#     sbl.set_scale(np.array([1,1,1])/math.sqrt(jnt_id))
#     jnt_pos = stem1.jlc.joints[jnt_id]['gl_posq']
#     sbl.set_pos(jnt_pos)
#     sbl.attach_to(base)


# main_stem.gen_stickmodel().attach_to(base)
base.run()
# main = jlc.JLChain()
# main.lnks[id]['collision_model'] = mgm.gen_
# main.gen_meshmodel()
# def gen_random_leaf():
#     rotmats = rm.gen_icorotmats(icolevel=2, rotation_interval=15, crop_angle=np.np.pi / 4,
#                                 toggle_flat=True)
#     return_rotmat = []
#     for rotmat in rotmats:
#         if rm.angle_between_vectors(rotmat[:,1], np.array([0,0,-1]))<np.pi:
#             return_rotmat.append(rotmat)
#
# leaf_rgba = [45 / 255, 90 / 255, 39 / 255, 1]
# stem_rgba = [97 / 255, 138 / 255, 61 / 255, 1]
#
# sb_leaf = mgm.GeometricModel(initializer="objects/soybean_leaf.stl")
# sb_leaf.set_rgba(rgba=leaf_rgba)
#
# stem0_spos = np.array([0, 0, 0])
# stem0_epos = np.array([0, 0, .05])
# main_stem = mgm.gen_stick(spos=stem0_spos, epos=stem0_epos, rgba=stem_rgba)
# main_stem.attach_to(base)
#
# sbl0 = sb_leaf.copy()
# l0_stem_spos = stem0_epos
# l0_stem_epos = stem0_epos + sbl0.get_rotmat()[:, 0] * .005
# l0_stem = mgm.gen_stick(spos=l0_stem_spos, epos=l0_stem_epos, rgba=stem_rgba)
# l0_stem.attach_to(base)
# sbl0.set_pos(l0_stem_epos)
# sbl0.attach_to(base)
#
# sbl1 = sb_leaf.copy()
# sbl1.set_pos(stem0_epos - np.array([0, 0, 0.005]))
# sbl1.set_rotmat(rm.rotmat_from_axangle([0, 0, 1], np.pi))
# sbl1.attach_to(base)

base.run()

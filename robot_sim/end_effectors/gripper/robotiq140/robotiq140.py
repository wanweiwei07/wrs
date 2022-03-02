import os
import math
import numpy as np

import basis.robot_math as rm
import modeling.geometric_model as gm
import modeling.model_collection as mc
import robot_sim._kinematics.jlchain as jl
import robot_sim.end_effectors.gripper.gripper_interface as gp


class Robotiq140(gp.GripperInterface):
    """
    author: kiyokawa, revised by weiwei
    date: 2020212
    """

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), cdmesh_type='box', name='robotiq140', enable_cc=True):

        super().__init__(pos=pos, rotmat=rotmat, cdmesh_type=cdmesh_type, name=name)
        this_dir, this_filename = os.path.split(__file__)
        cpl_end_pos = self.coupling.jnts[-1]['gl_posq']
        cpl_end_rotmat = self.coupling.jnts[-1]['gl_rotmatq']
        # - lft_outer
        self.lft_outer = jl.JLChain(pos=cpl_end_pos, rotmat=cpl_end_rotmat, homeconf=np.zeros(4), name='lft_outer')
        self.lft_outer.jnts[1]['loc_pos'] = np.array([0, -.0306011, .054904])
        self.lft_outer.jnts[1]['motion_rng'] = [.0, .7]
        self.lft_outer.jnts[1]['loc_rotmat'] = rm.rotmat_from_euler((math.pi / 2.0 + .725), 0, 0)
        self.lft_outer.jnts[1]['loc_motionax'] = np.array([-1, 0, 0])
        self.lft_outer.jnts[2]['loc_pos'] = np.array([0, 0.01821998610742, 0.0260018192872234])  # passive
        self.lft_outer.jnts[2]['loc_motionax'] = np.array([1, 0, 0])
        self.lft_outer.jnts[3]['loc_pos'] = np.array([0, 0.0817554015893473, -0.0282203446692936])
        self.lft_outer.jnts[3]['loc_rotmat'] = rm.rotmat_from_euler(-0.725, 0, 0)
        self.lft_outer.jnts[3]['loc_motionax'] = np.array([1, 0, 0])
        self.lft_outer.jnts[4]['loc_pos'] = np.array([0, 0.0420203446692936, -.03242])
        # - lft_inner
        self.lft_inner = jl.JLChain(pos=cpl_end_pos, rotmat=cpl_end_rotmat, homeconf=np.zeros(1), name='lft_inner')
        self.lft_inner.jnts[1]['loc_pos'] = np.array([0, -.0127, .06142])
        self.lft_inner.jnts[1]['loc_rotmat'] = rm.rotmat_from_euler((math.pi / 2.0 + .725), 0, 0)
        self.lft_inner.jnts[1]['loc_motionax'] = np.array([1, 0, 0])
        # - rgt_outer
        self.rgt_outer = jl.JLChain(pos=cpl_end_pos, rotmat=cpl_end_rotmat, homeconf=np.zeros(4), name='rgt_outer')
        self.rgt_outer.jnts[1]['loc_pos'] = np.array([0, .0306011, .054904])
        self.rgt_outer.jnts[1]['loc_rotmat'] = rm.rotmat_from_euler((math.pi / 2.0 + .725), 0, math.pi)
        self.rgt_outer.jnts[1]['loc_motionax'] = np.array([1, 0, 0])
        self.rgt_outer.jnts[2]['loc_pos'] = np.array([0, 0.01821998610742, 0.0260018192872234])  # passive
        self.rgt_outer.jnts[2]['loc_motionax'] = np.array([1, 0, 0])
        self.rgt_outer.jnts[3]['loc_pos'] = np.array([0, 0.0817554015893473, -0.0282203446692936])
        self.rgt_outer.jnts[3]['loc_rotmat'] = rm.rotmat_from_euler(-0.725, 0, 0)
        self.rgt_outer.jnts[3]['loc_motionax'] = np.array([1, 0, 0])
        self.rgt_outer.jnts[4]['loc_pos'] = np.array([0, 0.0420203446692936, -.03242])
        # - rgt_inner
        self.rgt_inner = jl.JLChain(pos=cpl_end_pos, rotmat=cpl_end_rotmat, homeconf=np.zeros(1), name='rgt_inner')
        self.rgt_inner.jnts[1]['loc_pos'] = np.array([0, 0.0127, 0.06142])
        self.rgt_inner.jnts[1]['loc_rotmat'] = rm.rotmat_from_euler((math.pi / 2.0 + .725), 0, math.pi)
        self.rgt_inner.jnts[1]['loc_motionax'] = np.array([1, 0, 0])
        # links
        # - lft_outer
        self.lft_outer.lnks[0]['name'] = "robotiq140_gripper_base"
        self.lft_outer.lnks[0]['loc_pos'] = np.zeros(3)
        self.lft_outer.lnks[0]['com'] = np.array([8.625e-08, -4.6583e-06, 0.03145])
        self.lft_outer.lnks[0]['mass'] = 0.22652
        self.lft_outer.lnks[0]['mesh_file'] = os.path.join(this_dir, "meshes", "robotiq_arg2f_base_link.stl")
        self.lft_outer.lnks[0]['rgba'] = [.2, .2, .2, 1]
        self.lft_outer.lnks[1]['name'] = "left_outer_knuckle"
        self.lft_outer.lnks[1]['loc_pos'] = np.zeros(3)
        self.lft_outer.lnks[1]['com'] = np.array([-0.000200000000003065, 0.0199435877845359, 0.0292245259211331])
        self.lft_outer.lnks[1]['mass'] = 0.00853198276973456
        self.lft_outer.lnks[1]['mesh_file'] = os.path.join(this_dir, "meshes", "robotiq_arg2f_140_outer_knuckle.stl")
        self.lft_outer.lnks[1]['rgba'] = [0.792156862745098, 0.819607843137255, 0.933333333333333, 1]
        self.lft_outer.lnks[2]['name'] = "left_outer_finger"
        self.lft_outer.lnks[2]['loc_pos'] = np.zeros(3)
        self.lft_outer.lnks[2]['com'] = np.array([0.00030115855001899, 0.0373907951953854, -0.0208027427000385])
        self.lft_outer.lnks[2]['mass'] = 0.022614240507152
        self.lft_outer.lnks[2]['mesh_file'] = os.path.join(this_dir, "meshes", "robotiq_arg2f_140_outer_finger.stl")
        self.lft_outer.lnks[2]['rgba'] = [.2, .2, .2, 1]
        self.lft_outer.lnks[3]['name'] = "left_inner_finger"
        self.lft_outer.lnks[3]['loc_pos'] = np.zeros(3)
        self.lft_outer.lnks[3]['com'] = np.array([0.000299999999999317, 0.0160078233491243, -0.0136945669206257])
        self.lft_outer.lnks[3]['mass'] = 0.0104003125914103
        self.lft_outer.lnks[3]['mesh_file'] = os.path.join(this_dir, "meshes", "robotiq_arg2f_140_inner_finger.stl")
        self.lft_outer.lnks[3]['rgba'] = [.2, .2, .2, 1]
        self.lft_outer.lnks[4]['name'] = "left_inner_finger_pad"
        self.lft_outer.lnks[4]['loc_pos'] = np.zeros(3)
        self.lft_outer.lnks[4]['mesh_file'] = os.path.join(this_dir, "meshes", "robotiq_arg2f_140_pad.stl")
        self.lft_outer.lnks[4]['scale'] = [1e-3, 1e-3, 1e-3]
        self.lft_outer.lnks[4]['rgba'] = [0.792156862745098, 0.819607843137255, 0.933333333333333, 1]
        # - lft_inner
        self.lft_inner.lnks[1]['name'] = "left_inner_knuckle"
        self.lft_inner.lnks[1]['loc_pos'] = np.zeros(3)
        self.lft_inner.lnks[1]['com'] = np.array([0.000123011831763771, 0.0507850843201817, 0.00103968640075166])
        self.lft_inner.lnks[1]['mass'] = 0.0271177346495152
        self.lft_inner.lnks[1]['mesh_file'] = os.path.join(this_dir, "meshes", "robotiq_arg2f_140_inner_knuckle.stl")
        self.lft_inner.lnks[1]['rgba'] = [.2, .2, .2, 1]
        # - rgt_outer
        self.rgt_outer.lnks[1]['name'] = "right_outer_knuckle"
        self.rgt_outer.lnks[1]['loc_pos'] = np.zeros(3)
        self.rgt_outer.lnks[1]['com'] = np.array([-0.000200000000003065, 0.0199435877845359, 0.0292245259211331])
        self.rgt_outer.lnks[1]['mass'] = 0.00853198276973456
        self.rgt_outer.lnks[1]['mesh_file'] = os.path.join(this_dir, "meshes", "robotiq_arg2f_140_outer_knuckle.stl")
        self.rgt_outer.lnks[1]['rgba'] = [0.792156862745098, 0.819607843137255, 0.933333333333333, 1]
        self.rgt_outer.lnks[2]['name'] = "right_outer_finger"
        self.rgt_outer.lnks[2]['loc_pos'] = np.zeros(3)
        self.rgt_outer.lnks[2]['com'] = np.array([0.00030115855001899, 0.0373907951953854, -0.0208027427000385])
        self.rgt_outer.lnks[2]['mass'] = 0.022614240507152
        self.rgt_outer.lnks[2]['mesh_file'] = os.path.join(this_dir, "meshes", "robotiq_arg2f_140_outer_finger.stl")
        self.rgt_outer.lnks[2]['rgba'] = [.2, .2, .2, 1]
        self.rgt_outer.lnks[3]['name'] = "right_inner_finger"
        self.rgt_outer.lnks[3]['loc_pos'] = np.zeros(3)
        self.rgt_outer.lnks[3]['com'] = np.array([0.000299999999999317, 0.0160078233491243, -0.0136945669206257])
        self.rgt_outer.lnks[3]['mass'] = 0.0104003125914103
        self.rgt_outer.lnks[3]['mesh_file'] = os.path.join(this_dir, "meshes", "robotiq_arg2f_140_inner_finger.stl")
        self.rgt_outer.lnks[3]['rgba'] = [.2, .2, .2, 1]
        self.rgt_outer.lnks[4]['name'] = "right_inner_finger_pad"
        self.rgt_outer.lnks[4]['loc_pos'] = np.zeros(3)
        self.rgt_outer.lnks[4]['mesh_file'] = os.path.join(this_dir, "meshes", "robotiq_arg2f_140_pad.stl")
        self.rgt_outer.lnks[4]['scale'] = [1e-3, 1e-3, 1e-3]
        self.rgt_outer.lnks[4]['rgba'] = [0.792156862745098, 0.819607843137255, 0.933333333333333, 1]
        # - rgt_inner
        self.rgt_inner.lnks[1]['name'] = "right_inner_knuckle"
        self.rgt_inner.lnks[1]['loc_pos'] = np.zeros(3)
        self.rgt_inner.lnks[1]['com'] = np.array([0.000123011831763771, 0.0507850843201817, 0.00103968640075166])
        self.rgt_inner.lnks[1]['mass'] = 0.0271177346495152
        self.rgt_inner.lnks[1]['mesh_file'] = os.path.join(this_dir, "meshes", "robotiq_arg2f_140_inner_knuckle.stl")
        self.rgt_inner.lnks[1]['rgba'] = [.2, .2, .2, 1]
        # reinitialize
        self.lft_outer.reinitialize()
        self.lft_inner.reinitialize()
        self.rgt_outer.reinitialize()
        self.rgt_inner.reinitialize()
        # jaw width
        self.jawwidth_rng = [0.0, .140]
        # jaw center
        self.jaw_center_pos = np.array([0, 0, .19])  # position for initial state (fully open)
        # relative jaw center pos
        self.jaw_center_pos_rel = self.jaw_center_pos - self.lft_outer.jnts[4]['gl_posq']
        # collision detection
        self.all_cdelements = []
        self.enable_cc(toggle_cdprimit=enable_cc)

    def enable_cc(self, toggle_cdprimit):
        if toggle_cdprimit:
            super().enable_cc()
            # cdprimit
            self.cc.add_cdlnks(self.lft_outer, [0, 1, 2, 3, 4])
            self.cc.add_cdlnks(self.lft_inner, [1])
            self.cc.add_cdlnks(self.rgt_outer, [1, 2, 3, 4])
            self.cc.add_cdlnks(self.rgt_inner, [1])
            activelist = [self.lft_outer.lnks[0],
                          self.lft_outer.lnks[1],
                          self.lft_outer.lnks[2],
                          self.lft_outer.lnks[3],
                          self.lft_outer.lnks[4],
                          self.lft_inner.lnks[1],
                          self.rgt_outer.lnks[1],
                          self.rgt_outer.lnks[2],
                          self.rgt_outer.lnks[3],
                          self.rgt_outer.lnks[4],
                          self.rgt_inner.lnks[1]]
            self.cc.set_active_cdlnks(activelist)
            self.all_cdelements = self.cc.all_cdelements
        # cdmesh
        for cdelement in self.all_cdelements:
            cdmesh = cdelement['collision_model'].copy()
            self.cdmesh_collection.add_cm(cdmesh)

    def fix_to(self, pos, rotmat, angle=None):
        self.pos = pos
        self.rotmat = rotmat
        if angle is not None:
            self.lft_outer.jnts[1]['motion_val'] = angle
            self.lft_outer.jnts[3]['motion_val'] = self.lft_outer.jnts[1]['motion_val']
            self.lft_inner.jnts[1]['motion_val'] = -self.lft_outer.jnts[1]['motion_val']
            self.rgt_outer.jnts[1]['motion_val'] = -self.lft_outer.jnts[1]['motion_val']
            self.rgt_outer.jnts[3]['motion_val'] = self.lft_outer.jnts[1]['motion_val']
            self.rgt_inner.jnts[1]['motion_val'] = -self.lft_outer.jnts[1]['motion_val']
        self.coupling.fix_to(self.pos, self.rotmat)
        cpl_end_pos = self.coupling.jnts[-1]['gl_posq']
        cpl_end_rotmat = self.coupling.jnts[-1]['gl_rotmatq']
        self.lft_outer.fix_to(cpl_end_pos, cpl_end_rotmat)
        self.lft_inner.fix_to(cpl_end_pos, cpl_end_rotmat)
        self.rgt_inner.fix_to(cpl_end_pos, cpl_end_rotmat)
        self.rgt_outer.fix_to(cpl_end_pos, cpl_end_rotmat)

    def fk(self, motion_val):
        """
        lft_outer is the only active joint, all others mimic this one
        :param: angle, radian
        """
        if self.lft_outer.jnts[1]['motion_rng'][0] <= motion_val <= self.lft_outer.jnts[1]['motion_rng'][1]:
            self.lft_outer.jnts[1]['motion_val'] = motion_val
            self.lft_outer.jnts[3]['motion_val'] = self.lft_outer.jnts[1]['motion_val']
            self.lft_inner.jnts[1]['motion_val'] = -self.lft_outer.jnts[1]['motion_val']
            self.rgt_outer.jnts[1]['motion_val'] = -self.lft_outer.jnts[1]['motion_val']
            self.rgt_outer.jnts[3]['motion_val'] = self.lft_outer.jnts[1]['motion_val']
            self.rgt_inner.jnts[1]['motion_val'] = -self.lft_outer.jnts[1]['motion_val']
            self.lft_outer.fk()
            self.lft_inner.fk()
            self.rgt_outer.fk()
            self.rgt_inner.fk()
        else:
            raise ValueError("The angle parameter is out of range!")

    def _from_distance_to_radians(self, distance):
        """
        private helper function to convert a command in meters to radians (joint value)
        """
        # return np.clip(
        #   self.lft_outer.jnts[1]['motion_rng'][1] - ((self.lft_outer.jnts[1]['motion_rng'][1]/self.jawwidth_rng[1]) * distance),
        #   self.lft_outer.jnts[1]['motion_rng'][0], self.lft_outer.jnts[1]['motion_rng'][1]) # kiyokawa, commented out by weiwei
        return np.clip(self.lft_outer.jnts[1]['motion_rng'][1] - math.asin(
            (math.sin(self.lft_outer.jnts[1]['motion_rng'][1]) / self.jawwidth_rng[1]) * distance),
                       self.lft_outer.jnts[1]['motion_rng'][0], self.lft_outer.jnts[1]['motion_rng'][1])

    def jaw_to(self, jaw_width):
        if jaw_width > self.jawwidth_rng[1]:
            raise ValueError(f"Jawwidth must be {self.jawwidth_rng[0]}mm~{self.jawwidth_rng[1]}mm!")
        motion_val = self._from_distance_to_radians(jaw_width)
        self.fk(motion_val)
        # TODO dynamically change jaw center
        # print(self.jaw_center_pos_rel)
        self.jaw_center_pos=np.array([0,0,self.lft_outer.jnts[4]['gl_posq'][2]+self.jaw_center_pos_rel[2]])

    def gen_stickmodel(self,
                       tcp_jntid=None,
                       tcp_loc_pos=None,
                       tcp_loc_rotmat=None,
                       toggle_tcpcs=False,
                       toggle_jntscs=False,
                       toggle_connjnt=False,
                       name='robotiq140_stickmodel'):
        sm_collection = mc.ModelCollection(name=name)
        self.coupling.gen_stickmodel(toggle_tcpcs=False,
                                     toggle_jntscs=toggle_jntscs).attach_to(sm_collection)
        self.lft_outer.gen_stickmodel(toggle_tcpcs=False,
                                      toggle_jntscs=toggle_jntscs,
                                      toggle_connjnt=toggle_connjnt).attach_to(sm_collection)
        self.lft_inner.gen_stickmodel(toggle_tcpcs=False,
                                      toggle_jntscs=toggle_jntscs,
                                      toggle_connjnt=toggle_connjnt).attach_to(sm_collection)
        self.rgt_outer.gen_stickmodel(toggle_tcpcs=False,
                                      toggle_jntscs=toggle_jntscs,
                                      toggle_connjnt=toggle_connjnt).attach_to(sm_collection)
        self.rgt_inner.gen_stickmodel(toggle_tcpcs=False,
                                      toggle_jntscs=toggle_jntscs,
                                      toggle_connjnt=toggle_connjnt).attach_to(sm_collection)
        if toggle_tcpcs:
            jaw_center_gl_pos = self.rotmat.dot(grpr.jaw_center_pos) + self.pos
            jaw_center_gl_rotmat = self.rotmat.dot(grpr.jaw_center_rotmat)
            gm.gen_dashstick(spos=self.pos,
                             epos=jaw_center_gl_pos,
                             thickness=.0062,
                             rgba=[.5, 0, 1, 1],
                             type="round").attach_to(sm_collection)
            gm.gen_mycframe(pos=jaw_center_gl_pos, rotmat=jaw_center_gl_rotmat).attach_to(sm_collection)
        return sm_collection

    def gen_meshmodel(self,
                      toggle_tcpcs=False,
                      toggle_jntscs=False,
                      rgba=None,
                      name='robotiq140_meshmodel'):
        mm_collection = mc.ModelCollection(name=name)
        self.coupling.gen_meshmodel(toggle_tcpcs=False,
                                    toggle_jntscs=toggle_jntscs,
                                    rgba=rgba).attach_to(mm_collection)
        self.lft_outer.gen_meshmodel(toggle_tcpcs=False,
                                     toggle_jntscs=toggle_jntscs,
                                     rgba=rgba).attach_to(mm_collection)
        self.lft_inner.gen_meshmodel(toggle_tcpcs=False,
                                     toggle_jntscs=toggle_jntscs,
                                     rgba=rgba).attach_to(mm_collection)
        self.rgt_outer.gen_meshmodel(toggle_tcpcs=False,
                                     toggle_jntscs=toggle_jntscs,
                                     rgba=rgba).attach_to(mm_collection)
        self.rgt_inner.gen_meshmodel(toggle_tcpcs=False,
                                     toggle_jntscs=toggle_jntscs,
                                     rgba=rgba).attach_to(mm_collection)
        if toggle_tcpcs:
            jaw_center_gl_pos = self.rotmat.dot(self.jaw_center_pos) + self.pos
            jaw_center_gl_rotmat = self.rotmat.dot(self.jaw_center_rotmat)
            gm.gen_dashstick(spos=self.pos,
                             epos=jaw_center_gl_pos,
                             thickness=.0062,
                             rgba=[.5, 0, 1, 1],
                             type="round").attach_to(mm_collection)
            gm.gen_mycframe(pos=jaw_center_gl_pos, rotmat=jaw_center_gl_rotmat).attach_to(mm_collection)
        return mm_collection


if __name__ == '__main__':
    import visualization.panda.world as wd

    base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0])
    gm.gen_frame().attach_to(base)
    grpr = Robotiq140(enable_cc=True)
    # grpr.cdmesh_type='convexhull'
    grpr.jaw_to(.1)
    grpr.gen_meshmodel(toggle_tcpcs=True, rgba=[.3, .3, .0, .5], toggle_jntscs=True).attach_to(base)
    base.run()

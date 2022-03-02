import os
import math
import numpy as np
import modeling.model_collection as mc
import modeling.geometric_model as gm
import robot_sim._kinematics.jlchain as jl
import basis.robot_math as rm
import robot_sim.end_effectors.gripper.gripper_interface as gp


class Robotiq85Gelsight(gp.GripperInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), cdmesh_type='box', name='robotiq85', enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, cdmesh_type=cdmesh_type, name=name)
        this_dir, this_filename = os.path.split(__file__)
        cpl_end_pos = self.coupling.jnts[-1]['gl_posq']
        cpl_end_rotmat = self.coupling.jnts[-1]['gl_rotmatq']
        # - lft_outer
        self.lft_outer = jl.JLChain(pos=cpl_end_pos, rotmat=cpl_end_rotmat, homeconf=np.zeros(4), name='lft_outer')
        self.lft_outer.jnts[1]['loc_pos'] = np.array([0, -.0306011, .054904])
        self.lft_outer.jnts[1]['motion_rng'] = [.0, .8]
        self.lft_outer.jnts[1]['loc_rotmat'] = rm.rotmat_from_euler(0, 0, math.pi)
        self.lft_outer.jnts[1]['loc_motionax'] = np.array([1, 0, 0])
        self.lft_outer.jnts[2]['loc_pos'] = np.array([0, .0315, -.0041])  # passive
        self.lft_outer.jnts[2]['loc_motionax'] = np.array([1, 0, 0])
        self.lft_outer.jnts[3]['loc_pos'] = np.array([0, .0061, .0471])
        self.lft_outer.jnts[3]['loc_motionax'] = np.array([1, 0, 0])
        self.lft_outer.jnts[4]['loc_pos'] = np.zeros(3)
        # https://github.com/Danfoa uses geometry instead of the dae mesh. The following coordiante is needed
        # self.lft_outer.jnts[4]['loc_pos'] = np.array([0, -0.0220203446692936, .03242])
        # - lft_inner
        self.lft_inner = jl.JLChain(pos=cpl_end_pos, rotmat=cpl_end_rotmat, homeconf=np.zeros(1), name='lft_inner')
        self.lft_inner.jnts[1]['loc_pos'] = np.array([0, -.0127, .06142])
        self.lft_inner.jnts[1]['loc_rotmat'] = rm.rotmat_from_euler(0, 0, math.pi)
        self.lft_inner.jnts[1]['loc_motionax'] = np.array([1, 0, 0])
        # - rgt_outer
        self.rgt_outer = jl.JLChain(pos=cpl_end_pos, rotmat=cpl_end_rotmat, homeconf=np.zeros(4), name='rgt_outer')
        self.rgt_outer.jnts[1]['loc_pos'] = np.array([0, .0306011, .054904])
        self.rgt_outer.jnts[1]['loc_motionax'] = np.array([1, 0, 0])
        self.rgt_outer.jnts[2]['loc_pos'] = np.array([0, .0315, -.0041])  # passive
        self.rgt_outer.jnts[2]['loc_motionax'] = np.array([1, 0, 0])
        self.rgt_outer.jnts[3]['loc_pos'] = np.array([0, .0061, .0471])
        self.rgt_outer.jnts[3]['loc_motionax'] = np.array([1, 0, 0])
        self.rgt_outer.jnts[4]['loc_pos'] = np.zeros(3)
        # https://github.com/Danfoa uses geometry instead of the dae mesh. The following coordiante is needed
        # self.rgt_outer.jnts[4]['loc_pos'] = np.array([0, -0.0220203446692936, .03242])
        # - rgt_inner
        self.rgt_inner = jl.JLChain(pos=cpl_end_pos, rotmat=cpl_end_rotmat, homeconf=np.zeros(1), name='rgt_inner')
        self.rgt_inner.jnts[1]['loc_pos'] = np.array([0, .0127, .06142])
        self.rgt_inner.jnts[1]['loc_motionax'] = np.array([1, 0, 0])
        # links
        # - lft_outer
        self.lft_outer.lnks[0]['name'] = "robotiq85_gripper_base"
        self.lft_outer.lnks[0]['loc_pos'] = np.zeros(3)
        self.lft_outer.lnks[0]['com'] = np.array([8.625e-08, -4.6583e-06, 0.03145])
        self.lft_outer.lnks[0]['mass'] = 0.22652
        self.lft_outer.lnks[0]['mesh_file'] = os.path.join(this_dir, "meshes", "robotiq_arg2f_85_base_link_cvt.stl")
        self.lft_outer.lnks[0]['rgba'] = [.2, .2, .2, 1]
        self.lft_outer.lnks[1]['name'] = "left_outer_knuckle"
        self.lft_outer.lnks[1]['loc_pos'] = np.zeros(3)
        self.lft_outer.lnks[1]['com'] = np.array([-0.000200000000003065, 0.0199435877845359, 0.0292245259211331])
        self.lft_outer.lnks[1]['mass'] = 0.00853198276973456
        self.lft_outer.lnks[1]['mesh_file'] = os.path.join(this_dir, "meshes", "robotiq_arg2f_85_outer_knuckle.stl")
        self.lft_outer.lnks[1]['rgba'] = [0.792156862745098, 0.819607843137255, 0.933333333333333, 1]
        self.lft_outer.lnks[2]['name'] = "left_outer_finger"
        self.lft_outer.lnks[2]['loc_pos'] = np.zeros(3)
        self.lft_outer.lnks[2]['com'] = np.array([0.00030115855001899, 0.0373907951953854, -0.0208027427000385])
        self.lft_outer.lnks[2]['mass'] = 0.022614240507152
        self.lft_outer.lnks[2]['mesh_file'] = os.path.join(this_dir, "meshes", "robotiq_arg2f_85_outer_finger_cvt.stl")
        self.lft_outer.lnks[2]['rgba'] = [.2, .2, .2, 1]
        self.lft_outer.lnks[3]['name'] = "left_inner_finger"
        self.lft_outer.lnks[3]['loc_pos'] = np.zeros(3)
        self.lft_outer.lnks[3]['com'] = np.array([0.000299999999999317, 0.0160078233491243, -0.0136945669206257])
        self.lft_outer.lnks[3]['mass'] = 0.0104003125914103
        self.lft_outer.lnks[3]['mesh_file'] = os.path.join(this_dir, "meshes",
                                                          "robotiq_85_inner_finger_gelsight_cvt.stl")
        self.lft_outer.lnks[3]['rgba'] = [0.57, 0.57, 0.57, 1]
        self.lft_outer.lnks[4]['name'] = "left_inner_finger_pad"
        self.lft_outer.lnks[4]['loc_pos'] = np.zeros(3)
        # self.lft_outer.lnks[4]['mesh_file'] = os.path.join(this_dir, "meshes", "robotiq_arg2f_85_pad.dae")
        # self.lft_outer.lnks[4]['scale'] = [1e-3, 1e-3, 1e-3]
        # self.lft_outer.lnks[4]['rgba'] = [0.792156862745098, 0.819607843137255, 0.933333333333333, 1]
        # - lft_inner
        self.lft_inner.lnks[1]['name'] = "left_inner_knuckle"
        self.lft_inner.lnks[1]['loc_pos'] = np.zeros(3)
        self.lft_inner.lnks[1]['com'] = np.array([0.000123011831763771, 0.0507850843201817, 0.00103968640075166])
        self.lft_inner.lnks[1]['mass'] = 0.0271177346495152
        self.lft_inner.lnks[1]['mesh_file'] = os.path.join(this_dir, "meshes", "robotiq_arg2f_85_inner_knuckle_cvt.stl")
        self.lft_inner.lnks[1]['rgba'] = [.2, .2, .2, 1]
        # - rgt_outer
        self.rgt_outer.lnks[1]['name'] = "left_outer_knuckle"
        self.rgt_outer.lnks[1]['loc_pos'] = np.zeros(3)
        self.rgt_outer.lnks[1]['com'] = np.array([-0.000200000000003065, 0.0199435877845359, 0.0292245259211331])
        self.rgt_outer.lnks[1]['mass'] = 0.00853198276973456
        self.rgt_outer.lnks[1]['mesh_file'] = os.path.join(this_dir, "meshes", "robotiq_arg2f_85_outer_knuckle.stl")
        self.rgt_outer.lnks[1]['rgba'] = [0.792156862745098, 0.819607843137255, 0.933333333333333, 1]
        self.rgt_outer.lnks[2]['name'] = "left_outer_finger"
        self.rgt_outer.lnks[2]['loc_pos'] = np.zeros(3)
        self.rgt_outer.lnks[2]['com'] = np.array([0.00030115855001899, 0.0373907951953854, -0.0208027427000385])
        self.rgt_outer.lnks[2]['mass'] = 0.022614240507152
        self.rgt_outer.lnks[2]['mesh_file'] = os.path.join(this_dir, "meshes", "robotiq_arg2f_85_outer_finger_cvt.stl")
        self.rgt_outer.lnks[2]['rgba'] = [.2, .2, .2, 1]
        self.rgt_outer.lnks[3]['name'] = "left_inner_finger"
        self.rgt_outer.lnks[3]['loc_pos'] = np.zeros(3)
        self.rgt_outer.lnks[3]['com'] = np.array([0.000299999999999317, 0.0160078233491243, -0.0136945669206257])
        self.rgt_outer.lnks[3]['mass'] = 0.0104003125914103
        self.rgt_outer.lnks[3]['mesh_file'] = os.path.join(this_dir, "meshes",
                                                          "robotiq_85_inner_finger_gelsight_cvt.stl")
        self.rgt_outer.lnks[3]['rgba'] = [0.57, 0.57, 0.57, 1]
        self.rgt_outer.lnks[4]['name'] = "left_inner_finger_pad"
        self.rgt_outer.lnks[4]['loc_pos'] = np.zeros(3)
        # self.rgt_outer.lnks[4]['mesh_file'] = os.path.join(this_dir, "meshes", "robotiq_arg2f_85_pad.dae")
        # self.rgt_outer.lnks[4]['scale'] = [1e-3, 1e-3, 1e-3]
        # self.rgt_outer.lnks[4]['rgba'] = [0.792156862745098, 0.819607843137255, 0.933333333333333, 1]
        # - rgt_inner
        self.rgt_inner.lnks[1]['name'] = "left_inner_knuckle"
        self.rgt_inner.lnks[1]['loc_pos'] = np.zeros(3)
        self.rgt_inner.lnks[1]['com'] = np.array([0.000123011831763771, 0.0507850843201817, 0.00103968640075166])
        self.rgt_inner.lnks[1]['mass'] = 0.0271177346495152
        self.rgt_inner.lnks[1]['mesh_file'] = os.path.join(this_dir, "meshes", "robotiq_arg2f_85_inner_knuckle_cvt.stl")
        self.rgt_inner.lnks[1]['rgba'] = [.2, .2, .2, 1]
        # reinitialize
        self.lft_outer.reinitialize()
        self.lft_inner.reinitialize()
        self.rgt_outer.reinitialize()
        self.rgt_inner.reinitialize()
        # jaw width
        self.jawwidth_rng = [0.0, .085]
        # jaw center
        self.jaw_center_pos = np.array([0, 0, .180])
        # collision detection
        self.all_cdelements = []
        self.enable_cc(toggle_cdprimit=enable_cc)

    def enable_cc(self, toggle_cdprimit):
        if toggle_cdprimit:
            super().enable_cc()
            # cdprimit
            self.cc.add_cdlnks(self.lft_outer, [0, 1, 2, 3])
            self.cc.add_cdlnks(self.lft_inner, [1])
            self.cc.add_cdlnks(self.rgt_outer, [1, 2, 3])
            self.cc.add_cdlnks(self.rgt_inner, [1])
            activelist = [self.lft_outer.lnks[0],
                          self.lft_outer.lnks[1],
                          self.lft_outer.lnks[2],
                          self.lft_outer.lnks[3],
                          self.lft_inner.lnks[1],
                          self.rgt_outer.lnks[1],
                          self.rgt_outer.lnks[2],
                          self.rgt_outer.lnks[3],
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
            self.lft_outer.jnts[3]['motion_val'] = -self.lft_outer.jnts[1]['motion_val']
            self.lft_inner.jnts[1]['motion_val'] = self.lft_outer.jnts[1]['motion_val']
            self.rgt_outer.jnts[1]['motion_val'] = self.lft_outer.jnts[1]['motion_val']
            self.rgt_outer.jnts[3]['motion_val'] = -self.lft_outer.jnts[1]['motion_val']
            self.rgt_inner.jnts[1]['motion_val'] = self.lft_outer.jnts[1]['motion_val']
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
            self.lft_outer.jnts[3]['motion_val'] = -self.lft_outer.jnts[1]['motion_val']
            self.lft_inner.jnts[1]['motion_val'] = self.lft_outer.jnts[1]['motion_val']
            self.rgt_outer.jnts[1]['motion_val'] = self.lft_outer.jnts[1]['motion_val']
            self.rgt_outer.jnts[3]['motion_val'] = -self.lft_outer.jnts[1]['motion_val']
            self.rgt_inner.jnts[1]['motion_val'] = self.lft_outer.jnts[1]['motion_val']
            self.lft_outer.fk()
            self.lft_inner.fk()
            self.rgt_outer.fk()
            self.rgt_inner.fk()
        else:
            raise ValueError("The angle parameter is out of range!")

    def jaw_to(self, jaw_width):
        jaw_width = jaw_width + .028
        if jaw_width > 0.085:
            jaw_width = 0.085
        if jaw_width > self.jawwidth_rng[1]:
            raise ValueError(f"Jawwidth must be {self.jawwidth_rng[0]}mm~{self.jawwidth_rng[1]}mm!")
        motion_val = math.asin((self.jawwidth_rng[1] / 2.0 + .0064 - .0306011) / 0.055) - math.asin(
            (jaw_width / 2.0 + .0064 - .0306011) / 0.055)
        self.fk(motion_val)
        # 20220113 matsuoka
        rot = math.asin(math.asin(((jaw_width / 2.0 + 0.0064) - 0.0127) / 0.05715))
        self.jaw_center_pos = np.array([0.0, 0.0, 0.06142]) + np.array([0.0, 0.0, math.cos(rot) * 0.05715]) + np.array(
            [0.0, 0.0, 0.06325144]) + self.coupling.jnts[-1]['loc_pos']

    def gen_stickmodel(self,
                       tcp_jntid=None,
                       tcp_loc_pos=None,
                       tcp_loc_rotmat=None,
                       toggle_tcpcs=False,
                       toggle_jntscs=False,
                       toggle_connjnt=False,
                       name='robotiq85_stickmodel'):
        sm_collection = mc.ModelCollection(name=name)
        self.coupling.gen_stickmodel(toggle_tcpcs=False,
                                     toggle_jntscs=toggle_jntscs).attach_to(sm_collection)
        self.lft_outer.gen_stickmodel(toggle_tcpcs=toggle_tcpcs,
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
                      name='robotiq85_meshmodel'):
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
    import modeling.geometric_model as gm

    base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0])
    gm.gen_frame().attach_to(base)
    grpr = Robotiq85Gelsight(enable_cc=True)
    grpr.cdmesh_type = 'convexhull'
    # grpr.fk(.0)
    grpr.jaw_to(.0)
    grpr.gen_meshmodel(toggle_tcpcs=True, rgba=[.3, .3, .0, .5]).attach_to(base)
    # grpr.gen_stickmodel(togglejntscs=False).attach_to(base)
    # grpr.fix_to(pos=np.array([0, .3, .2]), rotmat=rm.rotmat_from_axangle([1, 0, 0], math.pi / 6))
    # grpr.gen_meshmodel().attach_to(base)
    grpr.show_cdprimit()
    # grpr.show_cdmesh()
    base.run()

    # base = wd.World(cam_pos=[.5, .5, .5], lookat_pos=[0, 0, 0])
    # model = cm.CollisionModel("./meshes/robotiq_arg2f_85_pad.dae")
    # model.set_scale([1e-3, 1e-3, 1e-3])
    # model.attach_to(base)
    # # gm.gen_frame().attach_to(base)
    # model.show_cdmesh()
    # base.run()

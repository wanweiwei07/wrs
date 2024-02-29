import os
import math
import numpy as np
import modeling.model_collection as mc
import modeling.geometric_model as gm
import robot_sim._kinematics.jlchain as jl
import robot_sim.end_effectors.gripper.gripper_interface as gi


class XArmGripper(gi.GripperInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), cdmesh_type='box', name='xarm_gripper', enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, cdmesh_type=cdmesh_type, name=name)
        this_dir, this_filename = os.path.split(__file__)
        cpl_end_pos = self.coupling.jnts[-1]['gl_posq']
        cpl_end_rotmat = self.coupling.jnts[-1]['gl_rotmatq']
        # - lft_outer
        self.lft_outer = jl.JLChain(pos=cpl_end_pos, rotmat=cpl_end_rotmat, home_conf=np.zeros(2), name='jlc_lft_outer')
        self.lft_outer.jnts[1]['pos_in_loc_tcp'] = np.array([0, .035, .059098])
        self.lft_outer.jnts[1]['motion_range'] = [.0, .85]
        self.lft_outer.jnts[1]['loc_motionax'] = np.array([1, 0, 0])
        self.lft_outer.jnts[2]['pos_in_loc_tcp'] = np.array([0, .035465, .042039])  # passive
        self.lft_outer.jnts[2]['loc_motionax'] = np.array([-1, 0, 0])
        # - lft_inner
        self.lft_inner = jl.JLChain(pos=cpl_end_pos, rotmat=cpl_end_rotmat, home_conf=np.zeros(1), name='jlc_lft_inner')
        self.lft_inner.jnts[1]['pos_in_loc_tcp'] = np.array([0, .02, .074098])
        self.lft_inner.jnts[1]['loc_motionax'] = np.array([1, 0, 0])
        # - rgt_outer
        self.rgt_outer = jl.JLChain(pos=cpl_end_pos, rotmat=cpl_end_rotmat, home_conf=np.zeros(2), name='jlc_rgt_outer')
        self.rgt_outer.jnts[1]['pos_in_loc_tcp'] = np.array([0, -.035, .059098])
        self.rgt_outer.jnts[1]['loc_motionax'] = np.array([-1, 0, 0])
        self.rgt_outer.jnts[2]['pos_in_loc_tcp'] = np.array([0, -.035465, .042039])  # passive
        self.rgt_outer.jnts[2]['loc_motionax'] = np.array([1, 0, 0])
        # - rgt_inner
        self.rgt_inner = jl.JLChain(pos=cpl_end_pos, rotmat=cpl_end_rotmat, home_conf=np.zeros(1), name='jlc_rgt_inner')
        self.rgt_inner.jnts[1]['pos_in_loc_tcp'] = np.array([0, -.02, .074098])
        self.rgt_inner.jnts[1]['loc_motionax'] = np.array([-1, 0, 0])
        # links
        # - lft_outer
        self.lft_outer.lnks[0]['name'] = 'lnk_base'
        self.lft_outer.lnks[0]['pos_in_loc_tcp'] = np.zeros(3)
        self.lft_outer.lnks[0]['com'] = np.array([-0.00065489, -0.0018497, 0.048028])
        self.lft_outer.lnks[0]['mass'] = 0.5415
        self.lft_outer.lnks[0]['mesh_file'] = os.path.join(this_dir, "meshes", "base_link.stl")
        self.lft_outer.lnks[1]['name'] = 'lnk_left_outer_knuckle'
        self.lft_outer.lnks[1]['pos_in_loc_tcp'] = np.zeros(3)
        self.lft_outer.lnks[1]['com'] = np.array([2.9948e-14, 0.021559, 0.015181])
        self.lft_outer.lnks[1]['mass'] = 0.033618
        self.lft_outer.lnks[1]['mesh_file'] = os.path.join(this_dir, "meshes", "left_outer_knuckle.stl")
        self.lft_outer.lnks[1]['rgba'] = [.2, .2, .2, 1]
        self.lft_outer.lnks[2]['name'] = 'lnk_left_finger'
        self.lft_outer.lnks[2]['pos_in_loc_tcp'] = np.zeros(3)
        self.lft_outer.lnks[2]['com'] = np.array([-2.4536e-14, -0.016413, 0.029258])
        self.lft_outer.lnks[2]['mass'] = 0.048304
        self.lft_outer.lnks[2]['mesh_file'] = os.path.join(this_dir, "meshes", "left_finger.stl")
        self.lft_outer.lnks[2]['rgba'] = [.2, .2, .2, 1]
        # - lft_inner
        self.lft_inner.lnks[1]['name'] = 'lnk_left_inner_knuckle'
        self.lft_inner.lnks[1]['pos_in_loc_tcp'] = np.zeros(3)
        self.lft_inner.lnks[1]['com'] = np.array([2.9948e-14, 0.021559, 0.015181])
        self.lft_inner.lnks[1]['mass'] = 0.033618
        self.lft_inner.lnks[1]['mesh_file'] = os.path.join(this_dir, "meshes", "left_inner_knuckle.stl")
        self.lft_inner.lnks[1]['rgba'] = [.2, .2, .2, 1]
        # - rgt_outer
        self.rgt_outer.lnks[1]['name'] = 'lnk_right_outer_knuckle'
        self.rgt_outer.lnks[1]['pos_in_loc_tcp'] = np.zeros(3)
        self.rgt_outer.lnks[1]['com'] = np.array([-3.1669e-14, -0.021559, 0.015181])
        self.rgt_outer.lnks[1]['mass'] = 0.033618
        self.rgt_outer.lnks[1]['mesh_file'] = os.path.join(this_dir, "meshes", "right_outer_knuckle.stl")
        self.rgt_outer.lnks[1]['rgba'] = [.2, .2, .2, 1]
        self.rgt_outer.lnks[2]['name'] = 'lnk_right_finger'
        self.rgt_outer.lnks[2]['pos_in_loc_tcp'] = np.zeros(3)
        self.rgt_outer.lnks[2]['com'] = np.array([2.5618e-14, 0.016413, 0.029258])
        self.rgt_outer.lnks[2]['mass'] = 0.048304
        self.rgt_outer.lnks[2]['mesh_file'] = os.path.join(this_dir, "meshes", "right_finger.stl")
        self.rgt_outer.lnks[2]['rgba'] = [.2, .2, .2, 1]
        # - rgt_inner
        self.rgt_inner.lnks[1]['name'] = 'lnk_right_inner_knuckle'
        self.rgt_inner.lnks[1]['pos_in_loc_tcp'] = np.zeros(3)
        self.rgt_inner.lnks[1]['com'] = np.array([1.866e-06, -0.022047, 0.026133])
        self.rgt_inner.lnks[1]['mass'] = 0.023013
        self.rgt_inner.lnks[1]['mesh_file'] = os.path.join(this_dir, "meshes", "right_inner_knuckle.stl")
        self.rgt_inner.lnks[1]['rgba'] = [.2, .2, .2, 1]
        # reinitialize
        self.lft_outer.finalize()
        self.lft_inner.finalize()
        self.rgt_outer.finalize()
        self.rgt_inner.finalize()
        # jaw center
        self.jaw_center_pos = np.array([0, 0, .15])
        # jaw range
        self.jaw_range = [0.0, .085]
        # collision detection
        self.all_cdelements = []
        self.enable_cc(toggle_cdprimit=enable_cc)

    def enable_cc(self, toggle_cdprimit):
        if toggle_cdprimit:
            super().enable_cc()
            self.cc.add_cdlnks(self.lft_outer, [0, 1, 2])
            self.cc.add_cdlnks(self.rgt_outer, [1, 2])
            activelist = [self.lft_outer.lnks[0],
                          self.lft_outer.lnks[1],
                          self.lft_outer.lnks[2],
                          self.rgt_outer.lnks[1],
                          self.rgt_outer.lnks[2]]
            self.cc.set_active_cdlnks(activelist)
            self.all_cdelements = self.cc.cce_dict
        else:
            self.all_cdelements = [self.lft_outer.lnks[0],
                                   self.lft_outer.lnks[1],
                                   self.lft_outer.lnks[2],
                                   self.rgt_outer.lnks[1],
                                   self.rgt_outer.lnks[2]]
        # cdmesh
        for cdelement in self.all_cdelements:
            cdmesh = cdelement['collision_model'].copy()
            self.cdmesh_collection.add_cm(cdmesh)

    def fix_to(self, pos, rotmat, motion_val=None):
        if motion_val is not None:
            self.lft_outer.jnts[1]['motion_value'] = motion_val
            self.lft_outer.jnts[2]['motion_value'] = self.lft_outer.jnts[1]['motion_value']
            self.lft_inner.jnts[1]['motion_value'] = self.lft_outer.jnts[1]['motion_value']
            self.rgt_outer.jnts[1]['motion_value'] = self.lft_outer.jnts[1]['motion_value']
            self.rgt_outer.jnts[2]['motion_value'] = self.lft_outer.jnts[1]['motion_value']
            self.rgt_inner.jnts[1]['motion_value'] = self.lft_outer.jnts[1]['motion_value']
        self.pos = pos
        self.rotmat = rotmat
        self.coupling.fix_to(self.pos, self.rotmat)
        cpl_end_pos = self.coupling.jnts[-1]['gl_posq']
        cpl_end_rotmat = self.coupling.jnts[-1]['gl_rotmatq']
        self.lft_outer.fix_to(cpl_end_pos, cpl_end_rotmat)
        self.lft_inner.fix_to(cpl_end_pos, cpl_end_rotmat)
        self.rgt_outer.fix_to(cpl_end_pos, cpl_end_rotmat)
        self.rgt_inner.fix_to(cpl_end_pos, cpl_end_rotmat)

    def fk(self, motion_val):
        """
        lft_outer is the only active joint, all others mimic this one
        :param: motion_value, radian
        """
        if self.lft_outer.jnts[1]['motion_range'][0] <= motion_val <= self.lft_outer.jnts[1]['motion_range'][1]:
            self.lft_outer.jnts[1]['motion_value'] = motion_val
            self.lft_outer.jnts[2]['motion_value'] = self.lft_outer.jnts[1]['motion_value']
            self.lft_inner.jnts[1]['motion_value'] = self.lft_outer.jnts[1]['motion_value']
            self.rgt_outer.jnts[1]['motion_value'] = self.lft_outer.jnts[1]['motion_value']
            self.rgt_outer.jnts[2]['motion_value'] = self.lft_outer.jnts[1]['motion_value']
            self.rgt_inner.jnts[1]['motion_value'] = self.lft_outer.jnts[1]['motion_value']
            self.lft_outer.fk()
            self.lft_inner.fk()
            self.rgt_outer.fk()
            self.rgt_inner.fk()
        else:
            raise ValueError("The angle parameter is out of range!")

    def change_jaw_width(self, jaw_width):
        if jaw_width > 0.085:
            raise ValueError("jaw_width must be 0mm~85mm!")
        angle = .85 - math.asin(jaw_width / 2.0 / 0.055)
        if angle < 0:
            angle = 0
        self.fk(angle)

    def get_jaw_width(self):
        angle = self.lft_outer.jnts[1]['motion_value']
        return math.sin(.85 - angle) * 0.055 * 2.0

    def gen_stickmodel(self, toggle_tcp_frame=False, toggle_jnt_frames=False, name='ee_stickmodel'):
        mm_collection = mc.ModelCollection(name=name)
        self.lft_outer.gen_stickmodel(toggle_tcpcs=toggle_tcp_frame,
                                      toggle_jntscs=toggle_jnt_frames,
                                      toggle_connjnt=toggle_connjnt).attach_to(mm_collection)
        self.lft_inner.gen_stickmodel(toggle_tcpcs=False,
                                      toggle_jntscs=toggle_jnt_frames).attach_to(mm_collection)
        self.rgt_outer.gen_stickmodel(toggle_tcpcs=False,
                                      toggle_jntscs=toggle_jnt_frames).attach_to(mm_collection)
        self.rgt_inner.gen_stickmodel(toggle_tcpcs=False,
                                      toggle_jntscs=toggle_jnt_frames).attach_to(mm_collection)
        return mm_collection

    def gen_meshmodel(self,
                      tcp_jntid=None,
                      tcp_loc_pos=None,
                      tcp_loc_rotmat=None,
                      toggle_tcp_frame=False,
                      toggle_jnt_frames=False,
                      rgba=None,
                      name='xarm_gripper_meshmodel'):
        mm_collection = mc.ModelCollection(name=name)
        self.lft_outer.gen_mesh_model(toggle_tcpcs=False,
                                      toggle_jntscs=toggle_jnt_frames,
                                      rgba=rgba).attach_to(mm_collection)
        self.lft_inner.gen_mesh_model(toggle_tcpcs=False,
                                      toggle_jntscs=toggle_jnt_frames,
                                      rgba=rgba).attach_to(mm_collection)
        self.rgt_outer.gen_mesh_model(toggle_tcpcs=False,
                                      toggle_jntscs=toggle_jnt_frames,
                                      rgba=rgba).attach_to(mm_collection)
        self.rgt_inner.gen_mesh_model(toggle_tcpcs=False,
                                      toggle_jntscs=toggle_jnt_frames,
                                      rgba=rgba).attach_to(mm_collection)
        if toggle_tcp_frame:
            jaw_center_gl_pos = self.rotmat.dot(self.jaw_center_pos) + self.pos
            jaw_center_gl_rotmat = self.rotmat.dot(self.acting_center_rotmat)
            gm.gen_dashed_stick(spos=self.pos,
                                epos=jaw_center_gl_pos,
                                radius=.0062,
                                rgba=[.5, 0, 1, 1],
                                type="round").attach_to(mm_collection)
            gm.gen_myc_frame(pos=jaw_center_gl_pos, rotmat=jaw_center_gl_rotmat).attach_to(mm_collection)
        return mm_collection


if __name__ == '__main__':
    import visualization.panda.world as wd
    import modeling.geometric_model as gm

    base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])
    gm.gen_frame().attach_to(base)
    # for angle in np.linspace(0, .85, 8):
    #     xag = XArmGripper()
    #     xag.fk(angle)
    #     xag.gen_meshmodel().attach_to(base)
    xag = XArmGripper(enable_cc=True)
    xag.change_jaw_width(0.05)
    print(xag.get_jaw_width())
    model = xag.gen_meshmodel()
    model.attach_to(base)
    # xag.show_cdprimit()
    xag.cdmesh_type = 'convexhull'
    # xag.show_cdmesh()
    # xag.gen_stickmodel().attach_to(base)
    base.run()

import os
import numpy as np
import wrs.modeling.model_collection as mc
from wrs import robot_sim as jl, robot_sim as gp, modeling as gm


class TBMGripper(gp.GripperInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), cdmesh_type='box', name='tbm_gripper', enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, cdmesh_type=cdmesh_type, name=name)
        this_dir, this_filename = os.path.split(__file__)
        cpl_end_pos = self.coupling.jnts[-1]['gl_posq']
        cpl_end_rotmat = self.coupling.jnts[-1]['gl_rotmatq']
        # left finger
        self.lft_fgr = jl.JLChain(pos=cpl_end_pos, rotmat=cpl_end_rotmat, home_conf=np.zeros(1), name='lft_outer')
        self.lft_fgr.jnts[1]['loc_pos'] = np.array([0.113, 0, -.058])
        self.lft_fgr.jnts[1]['motion_range'] = [-.8, .8]
        self.lft_fgr.jnts[1]['loc_motionax'] = np.array([0, 1, 0])
        # right finger
        self.rgt_fgr = jl.JLChain(pos=cpl_end_pos, rotmat=cpl_end_rotmat, home_conf=np.zeros(1), name='lft_outer')
        self.rgt_fgr.jnts[1]['loc_pos'] = np.array([0.113, 0, 0.058])
        self.rgt_fgr.jnts[1]['motion_range'] = [-.8, .8]
        self.rgt_fgr.jnts[1]['loc_motionax'] = np.array([0, 1, 0])
        # links
        # palm and left finger
        self.lft_fgr.lnks[0]['name'] = "palm"
        self.lft_fgr.lnks[0]['loc_pos'] = np.zeros(3)
        self.lft_fgr.lnks[0]['mesh_file'] = os.path.join(this_dir, "meshes", "palm_r.stl")
        self.lft_fgr.lnks[0]['rgba'] = [.5, .5, .5, 1]
        self.lft_fgr.lnks[1]['name'] = "finger1"
        self.lft_fgr.lnks[1]['loc_pos'] = np.zeros(3)
        self.lft_fgr.lnks[1]['mesh_file'] = os.path.join(this_dir, "meshes", "finger1_r.stl")
        self.lft_fgr.lnks[1]['rgba'] = [0.792156862745098, 0.819607843137255, 0.933333333333333, 1]
        # right finger
        self.rgt_fgr.lnks[1]['name'] = "finger2"
        self.rgt_fgr.lnks[1]['loc_pos'] = np.zeros(3)
        self.rgt_fgr.lnks[1]['mesh_file'] = os.path.join(this_dir, "meshes", "finger2_r.stl")
        self.rgt_fgr.lnks[1]['rgba'] = [0.792156862745098, 0.819607843137255, 0.933333333333333, 1]
        # reinitialize
        self.lft_fgr.finalize()
        self.rgt_fgr.finalize()
        # jaw range
        self.jaw_range = [0, .5]
        # jaw center
        self.jaw_center_pos = np.array([.325, 0, 0])
        # collision detection
        self.all_cdelements = []
        self.enable_cc(toggle_cdprimit=enable_cc)

    def enable_cc(self, toggle_cdprimit):
        if toggle_cdprimit:
            super().enable_cc()
            # cdprimit
            self.cc.add_cdlnks(self.lft_fgr, [0, 1])
            self.cc.add_cdlnks(self.rgt_fgr, [1])
            activelist = [self.lft_fgr.lnks[0],
                          self.lft_fgr.lnks[1],
                          self.rgt_fgr.lnks[1]]
            self.cc.set_active_cdlnks(activelist)
            self.all_cdelements = self.cc.cce_dict
        # cdmesh
        for cdelement in self.all_cdelements:
            cdmesh = cdelement['collision_model'].copy()
            self.cdmesh_collection.add_cm(cdmesh)

    def fix_to(self, pos, rotmat, angle=None):
        self._pos = pos
        self._rotmat = rotmat
        if angle is not None:
            self.lft_fgr.jnts[1]['motion_value'] = angle
            self.rgt_fgr.jnts[1]['motion_value'] = -self.lft_fgr.jnts[1]['motion_value']
        self.coupling.fix_to(self._pos, self._rotmat)
        cpl_end_pos = self.coupling.jnts[-1]['gl_posq']
        cpl_end_rotmat = self.coupling.jnts[-1]['gl_rotmatq']
        self.lft_fgr.fix_to(cpl_end_pos, cpl_end_rotmat)
        self.rgt_fgr.fix_to(cpl_end_pos, cpl_end_rotmat)

    def fk(self, motion_val):
        """
        lft_outer is the only active joint, all others mimic this one
        :param: angle, radian
        """
        if self.lft_fgr.jnts[1]['motion_range'][0] <= motion_val <= self.lft_fgr.jnts[1]['motion_range'][1]:
            self.lft_fgr.jnts[1]['motion_value'] = motion_val
            self.rgt_fgr.jnts[1]['motion_value'] = -self.lft_fgr.jnts[1]['motion_value']
            self.lft_fgr.fk()
            self.rgt_fgr.fk()
        else:
            raise ValueError("The angle parameter is out of range!")

    def change_jaw_width(self, jaw_width):
        if jaw_width > self.jaw_range[1]:
            raise ValueError(f"Jawwidth must be {self.jaw_range[0]}mm~{self.jaw_range[1]}mm!")
        self.fk(jaw_width)

    def gen_stickmodel(self, toggle_tcp_frame=False, toggle_jnt_frames=False, name='ee_stickmodel'):
        sm_collection = mc.ModelCollection(name=name)
        self.coupling.gen_stickmodel(toggle_tcp_frame=False,
                                     toggle_jnt_frames=toggle_jnt_frames).attach_to(sm_collection)
        self.lft_fgr.gen_stickmodel(toggle_tcpcs=toggle_tcp_frame,
                                    toggle_jntscs=toggle_jnt_frames,
                                    toggle_connjnt=toggle_connjnt).attach_to(sm_collection)
        self.rgt_fgr.gen_stickmodel(toggle_tcpcs=toggle_tcp_frame,
                                    toggle_jntscs=toggle_jnt_frames,
                                    toggle_connjnt=toggle_connjnt).attach_to(sm_collection)
        if toggle_tcp_frame:
            jaw_center_gl_pos = self._rotmat.dot(grpr.jaw_center_pos) + self._pos
            jaw_center_gl_rotmat = self._rotmat.dot(grpr.loc_acting_center_rotmat)
            gm.gen_dashed_stick(spos=self._pos,
                                epos=jaw_center_gl_pos,
                                radius=.0062,
                                rgba=[.5, 0, 1, 1],
                                type="round").attach_to(sm_collection)
            gm.gen_myc_frame(pos=jaw_center_gl_pos, rotmat=jaw_center_gl_rotmat).attach_to(sm_collection)
        return sm_collection

    def gen_meshmodel(self,
                      toggle_tcp_frame=False,
                      toggle_jnt_frames=False,
                      rgba=None,
                      name='tbmg_meshmodel'):
        mm_collection = mc.ModelCollection(name=name)
        self.coupling.gen_mesh_model(toggle_tcpcs=False,
                                     toggle_jntscs=toggle_jnt_frames,
                                     rgba=rgba).attach_to(mm_collection)
        self.lft_fgr.gen_mesh_model(toggle_tcpcs=False,
                                    toggle_jntscs=toggle_jnt_frames,
                                    rgba=rgba).attach_to(mm_collection)
        self.rgt_fgr.gen_mesh_model(toggle_tcpcs=False,
                                    toggle_jntscs=toggle_jnt_frames,
                                    rgba=rgba).attach_to(mm_collection)
        if toggle_tcp_frame:
            jaw_center_gl_pos = self._rotmat.dot(grpr.jaw_center_pos) + self._pos
            jaw_center_gl_rotmat = self._rotmat.dot(grpr.loc_acting_center_rotmat)
            gm.gen_dashed_stick(spos=self._pos,
                                epos=jaw_center_gl_pos,
                                radius=.0062,
                                rgba=[.5, 0, 1, 1],
                                type="round").attach_to(mm_collection)
            gm.gen_myc_frame(pos=jaw_center_gl_pos, rotmat=jaw_center_gl_rotmat).attach_to(mm_collection)
        return mm_collection


if __name__ == '__main__':
    import wrs.visualization.panda.world as wd

    base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0])
    gm.gen_frame().attach_to(base)
    grpr = TBMGripper(enable_cc=True)
    grpr.cdmesh_type = 'convexhull'
    # grpr.fk(.0)
    grpr.change_jaw_width(0.5)
    grpr.gen_meshmodel(toggle_tcp_frame=True).attach_to(base)
    grpr.gen_stickmodel(toggle_jnt_frames=False).attach_to(base)
    # grpr.fix_to(pos=np.array([0, .3, .2]), rotmat=rm.rotmat_from_axangle([1, 0, 0], math.pi / 6))
    # grpr.gen_meshmodel().attach_to(base)
    # grpr.show_cdprimit()
    # grpr.show_cdmesh()
    base.run()

    # base = wd.World(cam_pos=[.5, .5, .5], lookat_pos=[0, 0, 0])
    # model = mcm.CollisionModel("./meshes/robotiq_arg2f_85_pad.dae")
    # model.set_scale([1e-3, 1e-3, 1e-3])
    # model.attach_to(base)
    # # mgm.gen_frame().attach_to(base)
    # model.show_cdmesh()
    # base.run()

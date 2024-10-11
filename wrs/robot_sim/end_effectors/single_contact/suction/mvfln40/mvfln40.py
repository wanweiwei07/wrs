import os
import numpy as np
import wrs.modeling.model_collection as mc
from wrs import basis as rm, robot_sim as jl, modeling as gm
import wrs.robot_sim.end_effectors.single_contact.single_contact_interface as si


class MVFLN40(si.SCInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), cdmesh_type='box', name='mvfln40', enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, cdmesh_type=cdmesh_type, name=name)
        this_dir, this_filename = os.path.split(__file__)
        cpl_end_pos = self.coupling.jnts[-1]['gl_posq']
        cpl_end_rotmat = self.coupling.jnts[-1]['gl_rotmatq']
        self.jlc = jl.JLChain(pos=cpl_end_pos, rotmat=cpl_end_rotmat, home_conf=np.zeros(0), name='mvfln40_jlc')
        self.jlc.jnts[1]['loc_pos'] = np.array([0, .0, .068])
        self.jlc.lnks[0]['name'] = "mvfln40"
        self.jlc.lnks[0]['loc_pos'] = np.zeros(3)
        self.jlc.lnks[0]['mesh_file'] = os.path.join(this_dir, "meshes", "mvfln40.stl")
        self.jlc.lnks[0]['rgba'] = [.55, .55, .55, 1]
        # reinitialize
        self.jlc.finalize()
        # suction center
        self.suction_center_pos = np.array([0, 0, .068])
        # collision detection
        self.all_cdelements = []
        self.enable_cc(toggle_cdprimit=enable_cc)

    def enable_cc(self, toggle_cdprimit):
        if toggle_cdprimit:
            super().enable_cc()
            # cdprimit
            self.cc.add_cdlnks(self.jlc, [0])
            activelist = [self.jlc.lnks[0]]
            self.cc.set_active_cdlnks(activelist)
            self.all_cdelements = self.cc.cce_dict
        # cdmesh
        for cdelement in self.all_cdelements:
            cdmesh = cdelement['collision_model'].copy()
            self.cdmesh_collection.add_cm(cdmesh)

    def fix_to(self, pos, rotmat):
        self._pos = pos
        self._rotmat = rotmat
        self.coupling.fix_to(self._pos, self._rotmat)
        cpl_end_pos = self.coupling.jnts[-1]['gl_posq']
        cpl_end_rotmat = self.coupling.jnts[-1]['gl_rotmatq']
        self.jlc.fix_to(cpl_end_pos, cpl_end_rotmat)

    def gen_stickmodel(self,
                       toggle_tcp_frame=False,
                       toggle_jnt_frame=False,
                       toggle_connjnt=False,
                       name='suction_stickmodel'):
        mm_collection = mc.ModelCollection(name=name)
        self.coupling.gen_stickmodel(toggle_tcp_frame=False, toggle_jnt_frames=toggle_jnt_frame).attach_to(mm_collection)
        self.jlc.gen_stickmodel(toggle_tcp_frame=False,
                                toggle_jnt_frame=toggle_jnt_frame,
                                toggle_connjnt=toggle_connjnt).attach_to(mm_collection)
        if toggle_tcp_frame:
            suction_center_gl_pos = self._rotmat.dot(self.suction_center_pos) + self._pos
            suction_center_gl_rotmat = self._rotmat.dot(self.contact_center_rotmat)
            gm.gen_dashed_stick(spos=self._pos,
                                epos=suction_center_gl_pos,
                                radius=.0062,
                                rgba=[.5, 0, 1, 1],
                                type="round").attach_to(mm_collection)
            gm.gen_myc_frame(pos=suction_center_gl_pos, rotmat=suction_center_gl_rotmat).attach_to(mm_collection)
        return mm_collection

    def gen_meshmodel(self,
                      toggle_tcp_frame=False,
                      toggle_jnt_frame=False,
                      rgba=None,
                      name='xarm_gripper_meshmodel'):
        mm_collection = mc.ModelCollection(name=name)
        self.coupling.gen_mesh_model(toggle_tcp_frame=False,
                                     toggle_jnt_frame=toggle_jnt_frame,
                                     rgba=rgba).attach_to(mm_collection)
        self.jlc.gen_mesh_model(toggle_tcp_frame=False,
                                toggle_jnt_frame=toggle_jnt_frame,
                                rgba=rgba).attach_to(mm_collection)
        if toggle_tcp_frame:
            suction_center_gl_pos = self._rotmat.dot(self.suction_center_pos) + self._pos
            suction_center_gl_rotmat = self._rotmat.dot(self.contact_center_rotmat)
            gm.gen_dashed_stick(spos=self._pos,
                                epos=suction_center_gl_pos,
                                radius=.0062,
                                rgba=[.5, 0, 1, 1],
                                type="round").attach_to(mm_collection)
            gm.gen_myc_frame(pos=suction_center_gl_pos, rotmat=suction_center_gl_rotmat).attach_to(mm_collection)
        return mm_collection


if __name__ == '__main__':
    import wrs.visualization.panda.world as wd

    base = wd.World(cam_pos=[.5, .5, .5], lookat_pos=[0, 0, 0])
    gm.gen_frame().attach_to(base)
    # for angle in np.linspace(0, .85, 8):
    #     grpr = Robotiq85()
    #     grpr.fk(angle)
    #     grpr.gen_meshmodel().attach_to(base)
    grpr = MVFLN40(enable_cc=True)
    grpr.gen_meshmodel(toggle_tcp_frame=True).attach_to(base)
    # grpr.gen_stickmodel(toggle_jnt_frames=False).attach_to(base)
    grpr.fix_to(pos=np.array([0, .3, .2]), rotmat=rm.rotmat_from_axangle([1, 0, 0], .05))
    grpr.gen_meshmodel().attach_to(base)
    grpr.show_cdmesh()
    grpr.show_cdprimit()
    base.run()

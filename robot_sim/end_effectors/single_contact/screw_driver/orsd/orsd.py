import os
import numpy as np
import modeling.model_collection as mc
import robot_sim._kinematics.jlchain as jl
import basis.robot_math as rm
import robot_sim.end_effectors.single_contact.single_contact_interface as si
import modeling.collision_model as cm


class ORSD(si.SCTInterface):
    """
     # orsd = OnRobot ScrewDriver
    author: weiwei
    date: 20230803
    """

    def __init__(self,
                 pos=np.zeros(3),
                 rotmat=np.eye(3),
                 coupling_offset_pos=np.zeros(3),
                 coupling_offset_rotmat=np.eye(3),
                 cdmesh_type='box',
                 name='onrobot_screwdriver',
                 enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, cdmesh_type=cdmesh_type, name=name)
        this_dir, this_filename = os.path.split(__file__)
        self.coupling.jnts[-1]['loc_pos'] = coupling_offset_pos
        self.coupling.jnts[-1]['loc_rotmat'] = coupling_offset_rotmat
        self.coupling.lnks[0]['rgba'] = np.array([.35, .35, .35, 1])
        self.coupling.lnks[0]['collision_model'] = cm.gen_stick(self.coupling.jnts[0]['loc_pos'],
                                                                self.coupling.jnts[-1]['loc_pos'],
                                                                thickness=0.07,
                                                                # rgba=[.35, .35, .35, 1], rgb will be overwritten
                                                                type='rect',
                                                                sections=36)
        self.coupling.reinitialize()
        cpl_end_pos = self.coupling.jnts[-1]['gl_posq']
        cpl_end_rotmat = self.coupling.jnts[-1]['gl_rotmatq']
        # jlc
        self.jlc = jl.JLChain(pos=cpl_end_pos, rotmat=cpl_end_rotmat, homeconf=np.zeros(0), name='orsd_jlc')
        self.jlc.jnts[1]['loc_pos'] = np.array([0.16855000, 0, 0.09509044])
        self.jlc.lnks[0]['name'] = "orsd"
        self.jlc.lnks[0]['loc_pos'] = np.zeros(3)
        self.jlc.lnks[0]['mesh_file'] = os.path.join(this_dir, "meshes", "or_screwdriver.stl")
        self.jlc.lnks[0]['rgba'] = [.55, .55, .55, 1]
        # reinitialize
        self.jlc.reinitialize()
        #  action center
        self.action_center_pos = self.coupling.jnts[-1]['loc_rotmat'] @ np.array([0.16855000, 0, 0.09509044]) + coupling_offset_pos
        self.action_center_rotmat = self.coupling.jnts[-1]['loc_rotmat']
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
            self.all_cdelements = self.cc.all_cdelements
        # cdmesh
        for cdelement in self.all_cdelements:
            cdmesh = cdelement['collision_model'].copy()
            self.cdmesh_collection.add_cm(cdmesh)

    def fix_to(self, pos, rotmat):
        self.pos = pos
        self.rotmat = rotmat
        self.coupling.fix_to(self.pos, self.rotmat)
        cpl_end_pos = self.coupling.jnts[-1]['gl_posq']
        cpl_end_rotmat = self.coupling.jnts[-1]['gl_rotmatq']
        self.jlc.fix_to(cpl_end_pos, cpl_end_rotmat)

    def gen_stickmodel(self,
                       toggle_tcpcs=False,
                       toggle_jntscs=False,
                       toggle_connjnt=False,
                       name='suction_stickmodel'):
        stick_model = mc.ModelCollection(name=name)
        self.coupling.gen_stickmodel(toggle_tcpcs=False,
                                     toggle_jntscs=toggle_jntscs).attach_to(stick_model)
        self.jlc.gen_stickmodel(toggle_tcpcs=False,
                                toggle_jntscs=toggle_jntscs,
                                toggle_connjnt=toggle_connjnt).attach_to(stick_model)
        if toggle_tcpcs:
            self._toggle_tcpcs(stick_model)
        return stick_model

    def gen_meshmodel(self,
                      toggle_tcpcs=False,
                      toggle_jntscs=False,
                      rgba=None,
                      name='xarm_gripper_meshmodel'):
        mesh_model = mc.ModelCollection(name=name)
        self.coupling.gen_meshmodel(toggle_tcpcs=False,
                                    toggle_jntscs=toggle_jntscs,
                                    rgba=rgba).attach_to(mesh_model)
        self.jlc.gen_meshmodel(toggle_tcpcs=False,
                               toggle_jntscs=toggle_jntscs,
                               rgba=rgba).attach_to(mesh_model)
        if toggle_tcpcs:
            self._toggle_tcpcs(mesh_model)
        return mesh_model


if __name__ == '__main__':
    import visualization.panda.world as wd
    import modeling.geometric_model as gm

    base = wd.World(cam_pos=[.5, .5, .5], lookat_pos=[0, 0, 0])
    gm.gen_frame().attach_to(base)
    # for angle in np.linspace(0, .85, 8):
    #     grpr = Robotiq85()
    #     grpr.fk(angle)
    #     grpr.gen_meshmodel().attach_to(base)
    grpr = ORSD(coupling_offset_pos=np.array([0, 0, 0.0145]), enable_cc=True)
    grpr.gen_meshmodel(toggle_tcpcs=True).attach_to(base)
    grpr.gen_stickmodel(toggle_jntscs=True).attach_to(base)
    grpr.fix_to(pos=np.array([0, .3, .2]), rotmat=rm.rotmat_from_axangle([1, 0, 0], .05))
    grpr.gen_meshmodel().attach_to(base)
    grpr.show_cdmesh()
    grpr.show_cdprimit()
    base.run()

import os
import numpy as np
import modeling.model_collection as mc
import robot_sim._kinematics.jlchain as jl
import basis.robot_math as rm
import robot_sim.end_effectors.gripper.gripper_interface as gp


class CobottaGripper(gp.GripperInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), cdmesh_type='box', name='robotiqhe', enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, cdmesh_type=cdmesh_type, name=name)
        this_dir, this_filename = os.path.split(__file__)
        cpl_end_pos = self.coupling.jnts[-1]['gl_posq']
        cpl_end_rotmat = self.coupling.jnts[-1]['gl_rotmatq']
        self.jlc = jl.JLChain(pos=cpl_end_pos, rotmat=cpl_end_rotmat, homeconf=np.zeros(2), name='base_lft_finger')
        self.jlc.jnts[1]['loc_pos'] = np.array([0, .0, .06])
        self.jlc.jnts[1]['type'] = 'prismatic'
        self.jlc.jnts[1]['motion_rng'] = [0, .015]
        self.jlc.jnts[1]['loc_motionax'] = np.array([0, 1, 0])
        self.jlc.jnts[2]['loc_pos'] = np.array([0, .0, .0])
        self.jlc.jnts[2]['type'] = 'prismatic'
        self.jlc.jnts[2]['loc_motionax'] = np.array([0, 1, 0])
        self.jlc.lnks[0]['name'] = "base"
        self.jlc.lnks[0]['loc_pos'] = np.zeros(3)
        self.jlc.lnks[0]['mesh_file'] = os.path.join(this_dir, "meshes", "gripper_base.dae")
        self.jlc.lnks[0]['rgba'] = [.35, .35, .35, 1]
        self.jlc.lnks[1]['name'] = "finger1"
        self.jlc.lnks[1]['loc_pos'] = np.array([0, 0, -.06])
        self.jlc.lnks[1]['mesh_file'] = os.path.join(this_dir, "meshes", "left_finger.dae")
        self.jlc.lnks[1]['rgba'] = [.5, .5, .5, 1]
        self.jlc.lnks[2]['name'] = "finger2"
        self.jlc.lnks[2]['loc_pos'] = np.array([0, 0, -.06])
        self.jlc.lnks[2]['mesh_file'] = os.path.join(this_dir, "meshes", "right_finger.dae")
        self.jlc.lnks[2]['rgba'] = [.5, .5, .5, 1]
        # jaw width
        self.jawwidth_rng = [0.0, .03]
        # jaw center
        self.jaw_center_pos = np.array([0,0,.05])
        # reinitialize
        self.jlc.reinitialize()
        # collision detection
        self.all_cdelements=[]
        self.enable_cc(toggle_cdprimit=enable_cc)

    def enable_cc(self, toggle_cdprimit):
        if toggle_cdprimit:
            super().enable_cc()
            # cdprimit
            self.cc.add_cdlnks(self.jlc, [0, 1, 2])
            activelist = [self.jlc.lnks[0],
                          self.jlc.lnks[1],
                          self.jlc.lnks[2]]
            self.cc.set_active_cdlnks(activelist)
            self.all_cdelements = self.cc.all_cdelements
        # cdmesh
        for cdelement in self.all_cdelements:
            cdmesh = cdelement['collision_model'].copy()
            self.cdmesh_collection.add_cm(cdmesh)

    def fix_to(self, pos, rotmat, jaw_width=None):
        self.pos = pos
        self.rotmat = rotmat
        if jaw_width is not None:
            side_jawwidth = jaw_width / 2.0
            if 0 <= side_jawwidth <= .015:
                self.jlc.jnts[1]['motion_val'] = side_jawwidth
                self.jlc.jnts[2]['motion_val'] = -jaw_width
            else:
                raise ValueError("The angle parameter is out of range!")
        self.coupling.fix_to(self.pos, self.rotmat)
        cpl_end_pos = self.coupling.jnts[-1]['gl_posq']
        cpl_end_rotmat = self.coupling.jnts[-1]['gl_rotmatq']
        self.jlc.fix_to(cpl_end_pos, cpl_end_rotmat)

    def jaw_to(self, jaw_width):
        if jaw_width > self.jawwidth_rng[1]:
            raise ValueError("The jaw_width parameter is out of range!")
        side_jawwidth = jaw_width / 2.0
        self.jlc.jnts[1]['motion_val'] = side_jawwidth
        self.jlc.jnts[2]['motion_val'] = -jaw_width
        self.jlc.fk()

    def get_jawwidth(self):
        return -self.jlc.jnts[2]['motion_val']

    def gen_stickmodel(self,
                       toggle_tcpcs=False,
                       toggle_jntscs=False,
                       toggle_connjnt=False,
                       name='xarm_gripper_stickmodel'):
        stickmodel = mc.ModelCollection(name=name)
        self.coupling.gen_stickmodel(toggle_tcpcs=False,
                                     toggle_jntscs=toggle_jntscs).attach_to(stickmodel)
        self.jlc.gen_stickmodel(toggle_tcpcs=False,
                                toggle_jntscs=toggle_jntscs,
                                toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        if toggle_tcpcs:
            jaw_center_gl_pos = self.rotmat.dot(self.jaw_center_pos)+self.pos
            jaw_center_gl_rotmat = self.rotmat.dot(self.jaw_center_rotmat)
            gm.gen_dashstick(spos=self.pos,
                             epos=jaw_center_gl_pos,
                             thickness=.0062,
                             rgba=[.5,0,1,1],
                             type="round").attach_to(stickmodel)
            gm.gen_mycframe(pos=jaw_center_gl_pos, rotmat=jaw_center_gl_rotmat).attach_to(stickmodel)
        return stickmodel

    def gen_meshmodel(self,
                      toggle_tcpcs=False,
                      toggle_jntscs=False,
                      rgba=None,
                      name='xarm_gripper_meshmodel'):
        meshmodel = mc.ModelCollection(name=name)
        self.coupling.gen_meshmodel(toggle_tcpcs=False,
                                    toggle_jntscs=toggle_jntscs,
                                    rgba=rgba).attach_to(meshmodel)
        self.jlc.gen_meshmodel(toggle_tcpcs=False,
                               toggle_jntscs=toggle_jntscs,
                               rgba=rgba).attach_to(meshmodel)
        if toggle_tcpcs:
            jaw_center_gl_pos = self.rotmat.dot(self.jaw_center_pos)+self.pos
            jaw_center_gl_rotmat = self.rotmat.dot(self.jaw_center_rotmat)
            gm.gen_dashstick(spos=self.pos,
                             epos=jaw_center_gl_pos,
                             thickness=.0062,
                             rgba=[.5,0,1,1],
                             type="round").attach_to(meshmodel)
            gm.gen_mycframe(pos=jaw_center_gl_pos, rotmat=jaw_center_gl_rotmat).attach_to(meshmodel)
        return meshmodel


if __name__ == '__main__':
    import visualization.panda.world as wd
    import modeling.geometric_model as gm

    base = wd.World(cam_pos=[.5, .5, .5], lookat_pos=[0, 0, 0])
    gm.gen_frame().attach_to(base)
    # for angle in np.linspace(0, .85, 8):
    #     grpr = Robotiq85()
    #     grpr.fk(angle)
    #     grpr.gen_meshmodel().attach_to(base)
    grpr = CobottaGripper(enable_cc=True)
    grpr.jaw_to(.013)
    grpr.gen_meshmodel(toggle_tcpcs=True).attach_to(base)
    # grpr.gen_stickmodel(toggle_jntscs=False).attach_to(base)
    grpr.fix_to(pos=np.array([0, .3, .2]), rotmat=rm.rotmat_from_axangle([1, 0, 0], .05))
    grpr.gen_meshmodel().attach_to(base)
    grpr.show_cdmesh()
    grpr.show_cdprimit()
    base.run()

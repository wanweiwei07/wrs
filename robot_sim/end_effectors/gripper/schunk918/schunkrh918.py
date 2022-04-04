import os
import math
import numpy as np
import modeling.model_collection as mc
import robot_sim._kinematics.jlchain as jl
import basis.robot_math as rm
import robot_sim.end_effectors.gripper.gripper_interface as gp
import modeling.geometric_model as gm


class SchunkRH918(gp.GripperInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), cdmesh_type='box', name='schunkrh918', enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, cdmesh_type=cdmesh_type, name=name)
        this_dir, this_filename = os.path.split(__file__)
        cpl_end_pos = self.coupling.jnts[-1]['gl_posq']
        cpl_end_rotmat = self.coupling.jnts[-1]['gl_rotmatq']
        # - lft
        self.lft = jl.JLChain(pos=cpl_end_pos, rotmat=cpl_end_rotmat, homeconf=np.zeros(2), name='base_lft_slider_finger')
        self.lft.jnts[1]['loc_pos'] = np.array([-.01, .04, .073])
        self.lft.jnts[1]['type'] = 'prismatic'
        self.lft.jnts[1]['motion_rng'] = [0, .025]
        self.lft.jnts[1]['loc_motionax'] = np.array([0, -1, 0])
        self.lft.lnks[0]['name'] = "base"
        self.lft.lnks[0]['loc_pos'] = np.zeros(3)
        self.lft.lnks[0]['mesh_file'] = os.path.join(this_dir, "meshes", "base.stl")
        self.lft.lnks[0]['rgba'] = [.2, .2, .2, 1]
        self.lft.lnks[1]['name'] = "slider1"
        self.lft.lnks[1]['mesh_file'] = os.path.join(this_dir, "meshes", "slider.stl")
        self.lft.lnks[1]['rgba'] = [.5, .5, .5, 1]
        self.lft.lnks[1]['loc_pos'] = np.zeros(3)
        self.lft.lnks[1]['loc_rotmat'] = rm.rotmat_from_euler(0, 0, -math.pi/2)
        self.lft.jnts[2]['loc_pos'] = np.array([.02, .008, 0])
        self.lft.lnks[2]['name'] = "finger1"
        self.lft.lnks[2]['mesh_file'] = os.path.join(this_dir, "meshes", "finger.stl")
        self.lft.lnks[2]['rgba'] = [.8, .8, .8, 1]
        self.lft.lnks[2]['loc_pos'] = np.zeros(3)
        self.lft.lnks[2]['loc_rotmat'] = rm.rotmat_from_euler(0, math.pi, math.pi/2)
        # - rgt
        self.rgt = jl.JLChain(pos=cpl_end_pos, rotmat=cpl_end_rotmat, homeconf=np.zeros(2), name='rgt_finger')
        self.rgt.jnts[1]['loc_pos'] = np.array([.01, -.04, .073])
        self.rgt.jnts[1]['type'] = 'prismatic'
        self.rgt.jnts[1]['loc_motionax'] = np.array([0, 1, 0])
        self.rgt.lnks[1]['name'] = "slider2"
        self.rgt.lnks[1]['mesh_file'] = os.path.join(this_dir, "meshes", "slider.stl")
        self.rgt.lnks[1]['rgba'] = [.5, .5, .5, 1]
        self.rgt.lnks[1]['loc_pos'] = np.zeros(3)
        self.rgt.lnks[1]['loc_rotmat'] = rm.rotmat_from_euler(0, 0, math.pi / 2)
        self.rgt.jnts[2]['loc_pos'] = np.array([-.02, -.008, 0])
        self.rgt.lnks[2]['name'] = "finger2"
        self.rgt.lnks[2]['mesh_file'] = os.path.join(this_dir, "meshes", "finger.stl")
        self.rgt.lnks[2]['rgba'] = [.8, .8, .8, 1]
        self.rgt.lnks[2]['loc_pos'] = np.zeros(3)
        self.rgt.lnks[2]['loc_rotmat'] = rm.rotmat_from_euler(0, math.pi, -math.pi / 2)
        # jaw width
        self.jawwidth_rng = [0.0, .05]
        # jaw center
        self.jaw_center_pos = np.array([0, 0, .145])
        # reinitialize
        self.lft.reinitialize()
        self.rgt.reinitialize()
        # collision detection
        self.all_cdelements=[]
        self.enable_cc(toggle_cdprimit=enable_cc)

    def enable_cc(self, toggle_cdprimit):
        if toggle_cdprimit:
            super().enable_cc()
            # cdprimit
            self.cc.add_cdlnks(self.lft, [0, 1, 2])
            self.cc.add_cdlnks(self.rgt, [1, 2])
            activelist = [self.lft.lnks[0],
                          self.lft.lnks[1],
                          self.lft.lnks[2],
                          self.rgt.lnks[1],
                          self.rgt.lnks[2]]
            self.cc.set_active_cdlnks(activelist)
            self.all_cdelements = self.cc.all_cdelements
        # cdmesh
        for cdelement in self.all_cdelements:
            cdmesh = cdelement['collision_model'].copy()
            self.cdmesh_collection.add_cm(cdmesh)

    def fix_to(self, pos, rotmat, jawwidth=None):
        self.pos = pos
        self.rotmat = rotmat
        if jawwidth is not None:
            side_jawwidth = (.05 - jawwidth) / 2.0
            if 0 <= side_jawwidth <= .025:
                self.lft.jnts[1]['motion_val'] = side_jawwidth;
                self.rgt.jnts[1]['motion_val'] = self.lft.jnts[1]['motion_val']
            else:
                raise ValueError("The angle parameter is out of range!")
        self.coupling.fix_to(self.pos, self.rotmat)
        cpl_end_pos = self.coupling.jnts[-1]['gl_posq']
        cpl_end_rotmat = self.coupling.jnts[-1]['gl_rotmatq']
        self.lft.fix_to(cpl_end_pos, cpl_end_rotmat)
        self.rgt.fix_to(cpl_end_pos, cpl_end_rotmat)

    def fk(self, motion_val):
        """
        lft_outer is the only active joint, all others mimic this one
        :param: angle, radian
        """
        if self.lft.jnts[1]['motion_rng'][0] <= motion_val <= self.lft.jnts[1]['motion_rng'][1]:
            self.lft.jnts[1]['motion_val'] = motion_val
            self.rgt.jnts[1]['motion_val'] = self.lft.jnts[1]['motion_val']
            self.lft.fk()
            self.rgt.fk()
        else:
            raise ValueError("The motion_val parameter is out of range!")

    def jaw_to(self, jawwidth):
        if jawwidth > .05:
            raise ValueError("The jaw_width parameter is out of range!")
        self.fk(motion_val=(0.05-jawwidth) / 2.0)

    def get_jawwidth(self):
        motion_val = self.lft.jnts[1]['motion_val']
        return self.jawwidth_rng[1] - motion_val*2.0

    def gen_stickmodel(self,
                       tcp_jntid=None,
                       tcp_loc_pos=None,
                       tcp_loc_rotmat=None,
                       toggle_tcpcs=False,
                       toggle_jntscs=False,
                       toggle_connjnt=False,
                       name='schunkrh918_stickmodel'):
        stickmodel = mc.ModelCollection(name=name)
        self.coupling.gen_stickmodel(toggle_tcpcs=False,
                                     toggle_jntscs=toggle_jntscs).attach_to(stickmodel)
        self.lft.gen_stickmodel(toggle_tcpcs=False,
                                toggle_jntscs=toggle_jntscs,
                                toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        self.rgt.gen_stickmodel(toggle_tcpcs=False,
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
                      name='schunkrh918_meshmodel'):
        meshmodel = mc.ModelCollection(name=name)
        self.coupling.gen_meshmodel(toggle_tcpcs=False,
                                    toggle_jntscs=toggle_jntscs,
                                    rgba=rgba).attach_to(meshmodel)
        self.lft.gen_meshmodel(toggle_tcpcs=False,
                               toggle_jntscs=toggle_jntscs,
                               rgba=rgba).attach_to(meshmodel)
        self.rgt.gen_meshmodel(toggle_tcpcs=False,
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

    base = wd.World(cam_pos=[.5, .5, .5], lookat_pos=[0, 0, 0])
    gm.gen_frame().attach_to(base)
    # for angle in np.linspace(0, .85, 8):
    #     grpr = Robotiq85()
    #     grpr.fk(angle)
    #     grpr.gen_meshmodel().attach_to(base)
    grpr = SchunkRH918(enable_cc=True)
    grpr.jaw_to(.03)
    print("jaw_width = ", grpr.get_jawwidth())
    grpr.gen_meshmodel().attach_to(base)
    # grpr.gen_stickmodel().attach_to(base)
    grpr.fix_to(pos=np.array([0, .3, .2]), rotmat=rm.rotmat_from_axangle([1, 0, 0], .05))
    grpr.gen_meshmodel(toggle_tcpcs=True).attach_to(base)
    grpr.show_cdmesh()
    grpr.show_cdprimit()
    base.run()

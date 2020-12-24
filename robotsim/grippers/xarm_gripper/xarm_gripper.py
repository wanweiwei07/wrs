import os
import math
import numpy as np
import modeling.modelcollection as mc
import robotsim._kinematics.jlchain as jl
import robotsim.grippers.gripper_interface as gi


class XArmGripper(gi.GripperInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name='xarm_gripper'):
        super().__init__(pos=pos, rotmat=rotmat, name=name)
        this_dir, this_filename = os.path.split(__file__)
        cpl_end_pos = self.coupling.jnts[-1]['gl_posq']
        cpl_end_rotmat = self.coupling.jnts[-1]['gl_rotmatq']
        # - lft_outer
        self.lft_outer = jl.JLChain(pos=cpl_end_pos,
                                    rotmat=cpl_end_rotmat,
                                    homeconf=np.zeros(2),
                                    name='jlc_lft_outer')
        self.lft_outer.jnts[1]['loc_pos'] = np.array([0, .035, .059098])
        self.lft_outer.jnts[1]['rngmin'] = .0
        self.lft_outer.jnts[1]['rngmax'] = .85  # TODO change min-max to a tuple
        self.lft_outer.jnts[1]['loc_motionax'] = np.array([1, 0, 0])
        self.lft_outer.jnts[2]['loc_pos'] = np.array([0, .035465, .042039])  # passive
        self.lft_outer.jnts[2]['rngmin'] = .0
        self.lft_outer.jnts[2]['rngmax'] = .85  # TODO change min-max to a tuple
        self.lft_outer.jnts[2]['loc_motionax'] = np.array([-1, 0, 0])
        # - lft_inner
        self.lft_inner = jl.JLChain(pos=cpl_end_pos,
                                    rotmat=cpl_end_rotmat,
                                    homeconf=np.zeros(1),
                                    name='jlc_lft_inner')
        self.lft_inner.jnts[1]['loc_pos'] = np.array([0, .02, .074098])
        self.lft_inner.jnts[1]['rngmin'] = .0
        self.lft_inner.jnts[1]['rngmax'] = .85  # TODO change min-max to a tuple
        self.lft_inner.jnts[1]['loc_motionax'] = np.array([1, 0, 0])
        # - rgt_outer
        self.rgt_outer = jl.JLChain(pos=cpl_end_pos,
                                    rotmat=cpl_end_rotmat,
                                    homeconf=np.zeros(2),
                                    name='jlc_rgt_outer')
        self.rgt_outer.jnts[1]['loc_pos'] = np.array([0, -.035, .059098])
        self.rgt_outer.jnts[1]['rngmin'] = .0
        self.rgt_outer.jnts[1]['rngmax'] = .85  # TODO change min-max to a tuple
        self.rgt_outer.jnts[1]['loc_motionax'] = np.array([-1, 0, 0])
        self.rgt_outer.jnts[2]['loc_pos'] = np.array([0, -.035465, .042039])  # passive
        self.rgt_outer.jnts[2]['rngmin'] = .0
        self.rgt_outer.jnts[2]['rngmax'] = .85  # TODO change min-max to a tuple
        self.rgt_outer.jnts[2]['loc_motionax'] = np.array([1, 0, 0])
        # - rgt_inner
        self.rgt_inner = jl.JLChain(pos=cpl_end_pos,
                                    rotmat=cpl_end_rotmat,
                                    homeconf=np.zeros(1),
                                    name='jlc_rgt_inner')
        self.rgt_inner.jnts[1]['loc_pos'] = np.array([0, -.02, .074098])
        self.rgt_inner.jnts[1]['rngmin'] = .0
        self.rgt_inner.jnts[1]['rngmax'] = .85  # TODO change min-max to a tuple
        self.rgt_inner.jnts[1]['loc_motionax'] = np.array([-1, 0, 0])
        # links
        # - lft_outer
        self.lft_outer.lnks[0]['name'] = 'lnk_base'
        self.lft_outer.lnks[0]['loc_pos'] = np.zeros(3)
        self.lft_outer.lnks[0]['com'] = np.array([-0.00065489, -0.0018497, 0.048028])
        self.lft_outer.lnks[0]['mass'] = 0.5415
        self.lft_outer.lnks[0]['meshfile'] = os.path.join(this_dir, "meshes", "base_link.stl")
        self.lft_outer.lnks[1]['name'] = 'lnk_left_outer_knuckle'
        self.lft_outer.lnks[1]['loc_pos'] = np.zeros(3)
        self.lft_outer.lnks[1]['com'] = np.array([2.9948e-14, 0.021559, 0.015181])
        self.lft_outer.lnks[1]['mass'] = 0.033618
        self.lft_outer.lnks[1]['meshfile'] = os.path.join(this_dir, "meshes", "left_outer_knuckle.stl")
        self.lft_outer.lnks[1]['rgba'] = [.2, .2, .2, 1]
        self.lft_outer.lnks[2]['name'] = 'lnk_left_finger'
        self.lft_outer.lnks[2]['loc_pos'] = np.zeros(3)
        self.lft_outer.lnks[2]['com'] = np.array([-2.4536e-14, -0.016413, 0.029258])
        self.lft_outer.lnks[2]['mass'] = 0.048304
        self.lft_outer.lnks[2]['meshfile'] = os.path.join(this_dir, "meshes", "left_finger.stl")
        self.lft_outer.lnks[2]['rgba'] = [.2, .2, .2, 1]
        # - lft_inner
        self.lft_inner.lnks[1]['name'] = 'lnk_left_inner_knuckle'
        self.lft_inner.lnks[1]['loc_pos'] = np.zeros(3)
        self.lft_inner.lnks[1]['com'] = np.array([2.9948e-14, 0.021559, 0.015181])
        self.lft_inner.lnks[1]['mass'] = 0.033618
        self.lft_inner.lnks[1]['meshfile'] = os.path.join(this_dir, "meshes", "left_inner_knuckle.stl")
        self.lft_inner.lnks[1]['rgba'] = [.2, .2, .2, 1]
        # - rgt_outer
        self.rgt_outer.lnks[1]['name'] = 'lnk_right_outer_knuckle'
        self.rgt_outer.lnks[1]['loc_pos'] = np.zeros(3)
        self.rgt_outer.lnks[1]['com'] = np.array([-3.1669e-14, -0.021559, 0.015181])
        self.rgt_outer.lnks[1]['mass'] = 0.033618
        self.rgt_outer.lnks[1]['meshfile'] = os.path.join(this_dir, "meshes", "right_outer_knuckle.stl")
        self.rgt_outer.lnks[1]['rgba'] = [.2, .2, .2, 1]
        self.rgt_outer.lnks[2]['name'] = 'lnk_right_finger'
        self.rgt_outer.lnks[2]['loc_pos'] = np.zeros(3)
        self.rgt_outer.lnks[2]['com'] = np.array([2.5618e-14, 0.016413, 0.029258])
        self.rgt_outer.lnks[2]['mass'] = 0.048304
        self.rgt_outer.lnks[2]['meshfile'] = os.path.join(this_dir, "meshes", "right_finger.stl")
        self.rgt_outer.lnks[2]['rgba'] = [.2, .2, .2, 1]
        # - rgt_inner
        self.rgt_inner.lnks[1]['name'] = 'lnk_right_inner_knuckle'
        self.rgt_inner.lnks[1]['loc_pos'] = np.zeros(3)
        self.rgt_inner.lnks[1]['com'] = np.array([1.866e-06, -0.022047, 0.026133])
        self.rgt_inner.lnks[1]['mass'] = 0.023013
        self.rgt_inner.lnks[1]['meshfile'] = os.path.join(this_dir, "meshes", "right_inner_knuckle.stl")
        self.rgt_inner.lnks[1]['rgba'] = [.2, .2, .2, 1]
        # reinitialize
        self.lft_outer.reinitialize()
        self.lft_inner.reinitialize()
        self.rgt_outer.reinitialize()
        self.rgt_inner.reinitialize()
        # disable the localcc of the inners and outers
        self.lft_outer.disable_localcc()
        self.lft_inner.disable_localcc()
        self.rgt_outer.disable_localcc()
        self.rgt_inner.disable_localcc()
        # collision detection
        self.cc.add_cdlnks(self.lft_outer, [0, 1, 2])
        self.cc.add_cdlnks(self.rgt_outer, [1, 2])
        activelist = [self.lft_outer.lnks[0],
                      self.lft_outer.lnks[1],
                      self.lft_outer.lnks[2],
                      self.rgt_outer.lnks[1],
                      self.rgt_outer.lnks[2]]
        self.cc.set_active_cdlnks(activelist)
        # cdmesh
        for cdelement in self.cc.all_cdelements:
            pos = cdelement['gl_pos']
            rotmat = cdelement['gl_rotmat']
            cdmesh = cdelement['collisionmodel'].copy()
            cdmesh.set_pos(pos)
            cdmesh.set_rotmat(rotmat)
            self.cdmesh_collection.add_cm(cdmesh)

    def fix_to(self, pos, rotmat):
        self.pos = pos
        self.rotmat = rotmat
        self.coupling.fix_to(self.pos, self.rotmat)
        cpl_end_pos = self.coupling.jnts[-1]['gl_posq']
        cpl_end_rotmat = self.coupling.jnts[-1]['gl_rotmatq']
        self.lft_outer.fix_to(cpl_end_pos, cpl_end_rotmat)
        self.lft_inner.fix_to(cpl_end_pos, cpl_end_rotmat)
        self.rgt_outer.fix_to(cpl_end_pos, cpl_end_rotmat)
        self.rgt_inner.fix_to(cpl_end_pos, cpl_end_rotmat)
        self.is_fk_updated = self.lft_outer.is_fk_updated

    def fk(self, motion_val):
        """
        lft_outer is the only active joint, all others mimic this one
        :param: motion_val, radian
        """
        if self.lft_outer.jnts[1]['rngmin'] <= motion_val <= self.lft_outer.jnts[1]['rngmax']:
            self.lft_outer.jnts[1]['motion_val'] = motion_val
            self.lft_outer.jnts[2]['motion_val'] = self.lft_outer.jnts[1]['motion_val']
            self.lft_inner.jnts[1]['motion_val'] = self.lft_outer.jnts[1]['motion_val']
            self.rgt_outer.jnts[1]['motion_val'] = self.lft_outer.jnts[1]['motion_val']
            self.rgt_outer.jnts[2]['motion_val'] = self.lft_outer.jnts[1]['motion_val']
            self.rgt_inner.jnts[1]['motion_val'] = self.lft_outer.jnts[1]['motion_val']
            self.lft_outer.fk()
            self.lft_inner.fk()
            self.rgt_outer.fk()
            self.rgt_inner.fk()
        else:
            raise ValueError("The angle parameter is out of range!")
        self.is_fk_updated = self.lft_outer.is_fk_updated

    def jaw_to(self, jawwidth):
        if jawwidth > 0.082:
            raise ValueError("Jawwidth must be 0mm~82mm!")
        angle = .85 - math.asin(jawwidth/2.0/0.055)
        self.fk(angle)

    def gen_stickmodel(self,
                       tcp_jntid=None,
                       tcp_loc_pos=None,
                       tcp_loc_rotmat=None,
                       toggle_tcpcs=False,
                       toggle_jntscs=False,
                       toggle_connjnt=False,
                       name='xarm_gripper_stickmodel'):
        stickmodel = mc.ModelCollection(name=name)
        self.lft_outer.gen_stickmodel(tcp_jntid=tcp_jntid,
                                      tcp_loc_pos=tcp_loc_pos,
                                      tcp_loc_rotmat=tcp_loc_rotmat,
                                      toggle_tcpcs=toggle_tcpcs,
                                      toggle_jntscs=toggle_jntscs,
                                      toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        self.lft_inner.gen_stickmodel(tcp_loc_pos=None,
                                      tcp_loc_rotmat=None,
                                      toggle_tcpcs=False,
                                      toggle_jntscs=toggle_jntscs).attach_to(stickmodel)
        self.rgt_outer.gen_stickmodel(tcp_loc_pos=None,
                                      tcp_loc_rotmat=None,
                                      toggle_tcpcs=False,
                                      toggle_jntscs=toggle_jntscs).attach_to(stickmodel)
        self.rgt_inner.gen_stickmodel(tcp_loc_pos=None,
                                      tcp_loc_rotmat=None,
                                      toggle_tcpcs=False,
                                      toggle_jntscs=toggle_jntscs).attach_to(stickmodel)
        return stickmodel

    def gen_meshmodel(self,
                      tcp_jntid=None,
                      tcp_loc_pos=None,
                      tcp_loc_rotmat=None,
                      toggle_tcpcs=False,
                      toggle_jntscs=False,
                      rgba=None,
                      name='xarm_gripper_meshmodel'):
        meshmodel = mc.ModelCollection(name=name)
        self.lft_outer.gen_meshmodel(tcp_jntid=tcp_jntid,
                                     tcp_loc_pos=tcp_loc_pos,
                                     tcp_loc_rotmat=tcp_loc_rotmat,
                                     toggle_tcpcs=toggle_tcpcs,
                                     toggle_jntscs=toggle_jntscs,
                                     rgba=rgba).attach_to(meshmodel)
        self.lft_inner.gen_meshmodel(tcp_loc_pos=None,
                                     tcp_loc_rotmat=None,
                                     toggle_tcpcs=False,
                                     toggle_jntscs=toggle_jntscs,
                                     rgba=rgba).attach_to(meshmodel)
        self.rgt_outer.gen_meshmodel(tcp_loc_pos=None,
                                     tcp_loc_rotmat=None,
                                     toggle_tcpcs=False,
                                     toggle_jntscs=toggle_jntscs,
                                     rgba=rgba).attach_to(meshmodel)
        self.rgt_inner.gen_meshmodel(tcp_loc_pos=None,
                                     tcp_loc_rotmat=None,
                                     toggle_tcpcs=False,
                                     toggle_jntscs=toggle_jntscs,
                                     rgba=rgba).attach_to(meshmodel)
        return meshmodel


if __name__ == '__main__':
    import visualization.panda.world as wd
    import modeling.geometricmodel as gm

    base = wd.World(campos=[2, 0, 1], lookatpos=[0, 0, 0])
    gm.gen_frame().attach_to(base)
    # for angle in np.linspace(0, .85, 8):
    #     xag = XArmGripper()
    #     xag.fk(angle)
    #     xag.gen_meshmodel().attach_to(base)
    xag = XArmGripper()
    xag.jaw_to(0.05)
    model = xag.gen_meshmodel(rgba=[.5,0,0,.3])
    model.attach_to(base)
    xag.show_cdprimit()
    xag.show_cdmesh()
    xag.gen_stickmodel().attach_to(base)
    base.run()

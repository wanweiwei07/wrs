#!/usr/bin/env python3

import os
import numpy as np

import basis.robot_math as rm
import modeling.model_collection as mc
import robot_sim._kinematics.jlchain as jl

import pyhiro.single_contact_device_interface as scdi


class SGB30(scdi.SingleContactDeviceInterface):

    def __init__(
            self,
            pos=np.zeros(3),
            rotmat=np.eye(3),
            cdmesh_type='box',
            name='sgb30',
            enable_cc=True):

        super().__init__(
            pos=pos,
            rotmat=rotmat,
            cdmesh_type=cdmesh_type,
            name=name)
        this_dir, this_filename = os.path.split(__file__)
        cpl_end_pos = self.coupling.jnts[-1]['gl_posq']
        cpl_end_rotmat = self.coupling.jnts[-1]['gl_rotmatq']
        self.jlc = jl.JLChain(
            pos=cpl_end_pos,
            rotmat=cpl_end_rotmat,
            homeconf=np.zeros(0),
            name='sgb30')
        self.jlc.jnts[1]['loc_pos'] = np.array([.0, .0, .0])
        self.jlc.lnks[0]['name'] = "sgb30"
        self.jlc.lnks[0]['loc_pos'] = np.array([.0, .0, .0])
        # self.jlc.lnks[0]['meshfile'] = os.path.join(
        #    this_dir, "meshes", "SGB.stl")
        self.jlc.lnks[0]['mesh_file'] = os.path.join(
            this_dir, "meshes", "sgb30m.stl")
        self.jlc.lnks[0]['rgba'] = [.55, .55, .55, 1]
        # reinitialize
        self.jlc.reinitialize()
        # suction center)
        # self.driver_center_pos = np.array([0, 0, .09])  # 0.8+0.1
        self.driver_center_pos = np.array([0., 0., 0.])  # 0.8+0.1 TODO
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

    def gen_stickmodel(
            self,
            toggle_tcpcs=False,
            toggle_jntscs=False,
            toggle_connjnt=False,
            name='driver_stickmodel'):
        mm_collection = mc.ModelCollection(name=name)
        self.coupling.gen_stickmodel(
            toggle_tcpcs=False,
            toggle_jntscs=toggle_jntscs).attach_to(mm_collection)
        self.jlc.gen_stickmodel(
            toggle_tcpcs=False,
            toggle_jntscs=toggle_jntscs,
            toggle_connjnt=toggle_connjnt).attach_to(mm_collection)
        if toggle_tcpcs:
            driver_center_gl_pos = \
                self.rotmat.dot(self.driver_center_pos) + self.pos
            driver_center_gl_rotmat = \
                self.rotmat.dot(self.driver_center_rotmat)
            gm.gen_dashstick(
                spos=self.pos,
                epos=driver_center_gl_pos,
                thickness=.0062,
                rgba=[.5, 0, 1, 1],
                type="round").attach_to(mm_collection)
            gm.gen_mycframe(
                pos=driver_center_gl_pos,
                rotmat=driver_center_gl_rotmat).attach_to(mm_collection)
        return mm_collection

    def gen_meshmodel(
            self,
            toggle_tcpcs=False,
            toggle_jntscs=False,
            rgba=None,
            name='xarm_gripper_meshmodel'):
        mm_collection = mc.ModelCollection(name=name)
        self.coupling.gen_meshmodel(
            toggle_tcpcs=False,
            toggle_jntscs=toggle_jntscs,
            rgba=rgba).attach_to(mm_collection)
        self.jlc.gen_meshmodel(
            toggle_tcpcs=False,
            toggle_jntscs=toggle_jntscs,
            rgba=rgba).attach_to(mm_collection)
        if toggle_tcpcs:
            driver_center_gl_pos = \
                self.rotmat.dot(self.driver_center_pos) + self.pos
            driver_center_gl_rotmat = \
                self.rotmat.dot(self.driver_center_rotmat)
            gm.gen_dashstick(
                spos=self.pos,
                epos=driver_center_gl_pos,
                thickness=.0062,
                rgba=[.5, 0, 1, 1],
                type="round").attach_to(mm_collection)
            gm.gen_mycframe(
                pos=driver_center_gl_pos,
                rotmat=driver_center_gl_rotmat).attach_to(mm_collection)
        return mm_collection


if __name__ == '__main__':
    import modeling.geometric_model as gm
    import visualization.panda.world as wd

    # base = wd.World(cam_pos=[.5, .5, .5], lookat_pos=[0, 0, 0])
    base = wd.World(
        cam_pos=[.55, .55, .55],
        lookat_pos=[.0, .1, .1],
        h=600, w=800)
    gm.gen_frame().attach_to(base)
    grpr = SGB30(enable_cc=True)
    grpr.gen_meshmodel(toggle_tcpcs=True).attach_to(base)
    # grpr.gen_stickmodel(toggle_jntscs=False).attach_to(base)
    grpr.fix_to(
        pos=np.array([0, .3, .2]),
        rotmat=rm.rotmat_from_axangle([1, 0, 0], .05))
    grpr.gen_meshmodel().attach_to(base)
    grpr.show_cdmesh()
    grpr.show_cdprimit()
    base.run()

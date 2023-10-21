"""
The simulation for the WRS gripper for nova2
Author: Chen Hao (chen960216@gmail.com), 20231009, osaka
"""
import os
import math
import numpy as np
import modeling.model_collection as mc
import modeling.collision_model as cm
from panda3d.core import CollisionNode, CollisionBox, Point3
import robot_sim._kinematics.jlchain as jl
import basis.robot_math as rm
import robot_sim.end_effectors.gripper.gripper_interface as gp


class Nova2HuriGripper(gp.GripperInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), cdmesh_type='convex_hull', name='nova2_huri_gripper2',
                 enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, cdmesh_type=cdmesh_type, name=name)
        this_dir, this_filename = os.path.split(__file__)
        cpl_end_pos = self.coupling.jnts[-1]['gl_posq']
        cpl_end_rotmat = self.coupling.jnts[-1]['gl_rotmatq']
        # gripper base
        self.body = jl.JLChain(pos=cpl_end_pos, rotmat=cpl_end_rotmat, homeconf=np.zeros(1), name='base')
        self.body.jnts[1]['loc_pos'] = np.array([0, 0, 0])
        self.body.lnks[0]['name'] = "base"
        self.body.lnks[0]['loc_pos'] = np.zeros(3)
        self.body.lnks[0]['collision_model'] = cm.CollisionModel(os.path.join(this_dir, "meshes", "base.stl"),
                                                                 cdprimit_type="user_defined",
                                                                 userdefined_cdprimitive_fn=self._base_cdnp,
                                                                 expand_radius=.001)
        self.body.lnks[0]['rgba'] = [.57, .57, .57, 1]

        # self.body.lnks[1]['name'] = "realsense_dual"
        # self.body.lnks[1]['loc_pos'] = np.zeros(3)
        # self.body.lnks[1]['collision_model'] = cm.CollisionModel(os.path.join(this_dir, "meshes", "dual_realsense.stl"),
        #                                                          expand_radius=.001)
        # self.body.lnks[1]['rgba'] = [.37, .37, .37, 1]

        # lft finger
        self.lft = jl.JLChain(pos=cpl_end_pos, rotmat=cpl_end_rotmat, homeconf=np.zeros(2), name='lft_finger')
        self.lft.jnts[1]['loc_pos'] = np.array([-0.02507, -0.0272, 0.018595])
        self.lft.jnts[1]['loc_rotmat'] = rm.rotmat_from_euler(0, 0, -math.pi)
        self.lft.jnts[1]['loc_motionax'] = np.array([0, -1, 0])
        self.lft.jnts[1]['motion_rng'] = [-math.pi, math.pi]
        self.lft.lnks[1]['name'] = "lft_finger_connector"
        self.lft.lnks[1]['collision_model'] = cm.CollisionModel(
            os.path.join(this_dir, "meshes", "connector.stl"), expand_radius=.001)
        self.lft.lnks[1]['rgba'] = [.65, .65, .65, 1]

        self.lft.jnts[2]['loc_pos'] = np.array([-0.02507, -0.0272, 0.077905])
        self.lft.jnts[2]['type'] = 'prismatic'
        self.lft.jnts[2]['loc_motionax'] = np.array([-1, 0, 0])
        self.lft.jnts[2]['motion_rng'] = [.0, 0.099]
        self.lft.lnks[2]['name'] = "lft_finger_link"
        self.lft.lnks[2]['collision_model'] = cm.CollisionModel(
            os.path.join(this_dir, "meshes", "finger.stl"), cdprimit_type="user_defined",
            userdefined_cdprimitive_fn=self._finger_cdnp, expand_radius=.001)
        self.lft.lnks[2]['rgba'] = [.65, .65, .65, 1]
        # # rgt finger
        self.rgt = jl.JLChain(pos=cpl_end_pos, rotmat=cpl_end_rotmat, homeconf=np.zeros(2), name='rgt_finger')
        self.rgt.jnts[1]['loc_pos'] = np.array([0.02507, 0.0272, 0.018595])
        self.rgt.jnts[1]['loc_motionax'] = np.array([0, 1, 0])
        self.rgt.jnts[1]['motion_rng'] = [-math.pi, math.pi]
        self.rgt.lnks[1]['name'] = "rgt_finger_connector"
        self.rgt.lnks[1]['collision_model'] = cm.CollisionModel(
            os.path.join(this_dir, "meshes", "connector.stl"), expand_radius=.001)
        self.rgt.lnks[1]['rgba'] = [.65, .65, .65, 1]

        self.rgt.jnts[2]['loc_pos'] = np.array([-0.02507, -0.0272, 0.077905])
        self.rgt.jnts[2]['type'] = 'prismatic'
        self.rgt.jnts[2]['loc_motionax'] = np.array([-1, 0, 0])
        self.rgt.jnts[2]['motion_rng'] = [.0, 0.099]
        self.rgt.lnks[2]['name'] = "rgt_finger_link"
        self.rgt.lnks[2]['collision_model'] = cm.CollisionModel(
            os.path.join(this_dir, "meshes", "finger.stl"), cdprimit_type="user_defined",
            userdefined_cdprimitive_fn=self._finger_cdnp, expand_radius=.001)
        self.rgt.lnks[2]['rgba'] = [.65, .65, .65, 1]

        # # reinitialize
        self.body.reinitialize(cdmesh_type=cdmesh_type)
        self.lft.reinitialize(cdmesh_type=cdmesh_type)
        self.rgt.reinitialize(cdmesh_type=cdmesh_type)
        # jaw width
        self.jawwidth_rng = [0.0, .198]
        # jaw center
        self.jaw_center_pos = np.array([0, 0, 0.225])
        # collision detection
        self.all_cdelements = []
        self.enable_cc(toggle_cdprimit=enable_cc)

    @staticmethod
    def _finger_cdnp(name, radius):
        collision_node = CollisionNode(name)
        collision_primitive_c0 = CollisionBox(Point3(0.021 - 0.03, 0.002 + 0.0125, -.02),
                                              x=0.03 + radius, y=0.0125 + radius, z=.02 + radius)
        collision_node.addSolid(collision_primitive_c0)
        collision_primitive_c1 = CollisionBox(Point3(0.0125, 0, 0.0125),
                                              x=0.0125 + radius, y=0.025 + radius, z=0.0125 + radius)
        collision_node.addSolid(collision_primitive_c1)
        collision_primitive_c2 = CollisionBox(Point3(0.011, 0, 0.0755),
                                              x=0.011 + radius, y=0.015 + radius, z=0.0725 + radius)
        collision_node.addSolid(collision_primitive_c2)
        return collision_node

    @staticmethod
    def _base_cdnp(name, radius):
        collision_node = CollisionNode(name)
        collision_primitive_c0 = CollisionBox(Point3(0, 0, 0.04325),
                                              x=0.04 + radius, y=0.0272 + radius, z=0.04325 + radius)
        collision_node.addSolid(collision_primitive_c0)
        collision_primitive_c1 = CollisionBox(Point3(0, -0.006 - 0.026, 0.108),
                                              x=0.026 + radius, y=0.026 + radius, z=0.0115+ radius)
        collision_node.addSolid(collision_primitive_c1)

        return collision_node

    def enable_cc(self, toggle_cdprimit):
        if toggle_cdprimit:
            super().enable_cc()
            # cdprimit
            # self.cc.add_cdlnks(self.body, [0, 1])
            self.cc.add_cdlnks(self.body, [0, ])
            self.cc.add_cdlnks(self.lft, [1, 2])
            self.cc.add_cdlnks(self.rgt, [1, 2])
            activelist = [self.body.lnks[0],
                          # self.body.lnks[1],
                          self.lft.lnks[1],
                          self.rgt.lnks[1],
                          self.lft.lnks[2],
                          self.rgt.lnks[2]
                          ]
            self.cc.set_active_cdlnks(activelist)
            self.all_cdelements = self.cc.all_cdelements
        else:
            self.all_cdelements = [self.body.lnks[0],
                                   # self.body.lnks[1],
                                   self.lft.lnks[1],
                                   self.rgt.lnks[1],
                                   self.lft.lnks[2],
                                   self.rgt.lnks[2]
                                   ]
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
        self.body.fix_to(cpl_end_pos, cpl_end_rotmat)
        self.lft.fix_to(cpl_end_pos, cpl_end_rotmat)
        self.rgt.fix_to(cpl_end_pos, cpl_end_rotmat)

    def fk(self, motion_val):
        """
        lft_outer is the only active joint, all others mimic this one
        :param: motion_val, meter or radian
        """
        if self.lft.jnts[2]['motion_rng'][0] <= -motion_val <= self.lft.jnts[2]['motion_rng'][1]:
            self.lft.jnts[2]['motion_val'] = motion_val
            self.rgt.jnts[2]['motion_val'] = self.lft.jnts[2]['motion_val']
            self.lft.fk()
            self.rgt.fk()
        else:
            raise ValueError("The motion_val parameter is out of range!")

    def jaw_to(self, jaw_width):
        if jaw_width > self.jawwidth_rng[1]:
            raise ValueError("The jaw_width parameter is out of range!")
        self.fk(motion_val=-jaw_width / 2.0)

    def get_jawwidth(self):
        return -self.lft.jnts[1]['motion_val'] * 2

    def gen_stickmodel(self,
                       tcp_jnt_id=None,
                       tcp_loc_pos=None,
                       tcp_loc_rotmat=None,
                       toggle_tcpcs=False,
                       toggle_jntscs=False,
                       toggle_connjnt=False,
                       name='lite6wrs_gripper_stickmodel'):
        stickmodel = mc.ModelCollection(name=name)
        self.coupling.gen_stickmodel(tcp_loc_pos=None,
                                     tcp_loc_rotmat=None,
                                     toggle_tcpcs=False,
                                     toggle_jntscs=toggle_jntscs).attach_to(stickmodel)
        self.body.gen_stickmodel(tcp_jnt_id=tcp_jnt_id,
                                 tcp_loc_pos=tcp_loc_pos,
                                 tcp_loc_rotmat=tcp_loc_rotmat,
                                 toggle_tcpcs=False,
                                 toggle_jntscs=toggle_jntscs,
                                 toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        self.lft.gen_stickmodel(tcp_jnt_id=tcp_jnt_id,
                                tcp_loc_pos=tcp_loc_pos,
                                tcp_loc_rotmat=tcp_loc_rotmat,
                                toggle_tcpcs=False,
                                toggle_jntscs=toggle_jntscs,
                                toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        self.rgt.gen_stickmodel(tcp_loc_pos=None,
                                tcp_loc_rotmat=None,
                                toggle_tcpcs=False,
                                toggle_jntscs=toggle_jntscs,
                                toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        if toggle_tcpcs:
            jaw_center_gl_pos = self.rotmat.dot(self.jaw_center_pos) + self.pos
            jaw_center_gl_rotmat = self.rotmat.dot(self.jaw_center_rotmat)
            gm.gen_dashstick(spos=self.pos,
                             epos=jaw_center_gl_pos,
                             thickness=.0062,
                             rgba=[.5, 0, 1, 1],
                             type="round").attach_to(stickmodel)
            gm.gen_mycframe(pos=jaw_center_gl_pos, rotmat=jaw_center_gl_rotmat).attach_to(stickmodel)

        return stickmodel

    def gen_meshmodel(self,
                      tcp_jnt_id=None,
                      tcp_loc_pos=None,
                      tcp_loc_rotmat=None,
                      toggle_tcpcs=False,
                      toggle_jntscs=False,
                      rgba=None,
                      name='nova2huri_gripper_meshmodel'):
        meshmodel = mc.ModelCollection(name=name)
        self.coupling.gen_meshmodel(tcp_loc_pos=None,
                                    tcp_loc_rotmat=None,
                                    toggle_tcpcs=False,
                                    toggle_jntscs=toggle_jntscs,
                                    rgba=rgba).attach_to(meshmodel)
        self.body.gen_meshmodel(tcp_jnt_id=tcp_jnt_id,
                                tcp_loc_pos=tcp_loc_pos,
                                tcp_loc_rotmat=tcp_loc_rotmat,
                                toggle_tcpcs=False,
                                toggle_jntscs=toggle_jntscs,
                                rgba=rgba).attach_to(meshmodel)
        self.lft.gen_meshmodel(tcp_jnt_id=tcp_jnt_id,
                               tcp_loc_pos=tcp_loc_pos,
                               tcp_loc_rotmat=tcp_loc_rotmat,
                               toggle_tcpcs=False,
                               toggle_jntscs=toggle_jntscs,
                               rgba=rgba).attach_to(meshmodel)
        self.rgt.gen_meshmodel(tcp_loc_pos=None,
                               tcp_loc_rotmat=None,
                               toggle_tcpcs=False,
                               toggle_jntscs=toggle_jntscs,
                               rgba=rgba).attach_to(meshmodel)
        if toggle_tcpcs:
            jaw_center_gl_pos = self.rotmat.dot(self.jaw_center_pos) + self.pos
            jaw_center_gl_rotmat = self.rotmat.dot(self.jaw_center_rotmat)
            gm.gen_dashstick(spos=self.pos,
                             epos=jaw_center_gl_pos,
                             thickness=.0062,
                             rgba=[.5, 0, 1, 1],
                             type="round").attach_to(meshmodel)
            gm.gen_mycframe(pos=jaw_center_gl_pos, rotmat=jaw_center_gl_rotmat).attach_to(meshmodel)
        return meshmodel


if __name__ == '__main__':
    import visualization.panda.world as wd
    import modeling.geometric_model as gm
    import robot_sim.end_effectors.gripper.lite6_wrs_gripper as gp1

    base = wd.World(cam_pos=[.5, .5, .5], lookat_pos=[0, 0, 0], auto_cam_rotate=False)
    gm.gen_frame().attach_to(base)
    # cm.CollisionModel("meshes/dual_realsense.stl", expand_radius=.001).attach_to(base)
    grpr = Nova2HuriGripper(enable_cc=True)
    grpr.jaw_to(.1)
    grpr.gen_meshmodel(toggle_tcpcs=True).attach_to(base)
    grpr.show_cdprimit()
    base.run()

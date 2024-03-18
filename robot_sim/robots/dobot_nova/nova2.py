"""
Simulation for the Nova2 With the WRS gripper
Author: Chen Hao (chen960216@gmail.com), 20231017, osaka
"""
import os
import numpy as np
from panda3d.core import CollisionNode, CollisionBox, Point3
import modeling.collision_model as cm
import modeling.model_collection as mc
import robot_sim._kinematics.jlchain as jl
from robot_sim.manipulators.dobot_nova2 import Nova2
from robot_sim.end_effectors.gripper.nova2_gripper import Nova2HuriGripper
import robot_sim.robots.single_arm_robot_interface as ri


class Nova2WRS(ri.SglArmRobotInterface):
    @staticmethod
    def _table_cdnp(name, radius):
        collision_node = CollisionNode(name)
        collision_primitive_r0 = CollisionBox(Point3(.27, 0, -.355),
                                              x=.36 + radius, y=.6 + radius, z=.355 + radius)
        collision_node.addSolid(collision_primitive_r0)
        collision_primitive_r1 = CollisionBox(Point3(-.06, -0.007, .325),
                                              x=.03 + radius, y=.03 + radius, z=.325 + radius)
        collision_node.addSolid(collision_primitive_r1)
        collision_primitive_r2 = CollisionBox(Point3(-.06, -0.15, .325),
                                              x=.03 + radius, y=.03 + radius, z=.325 + radius)
        collision_node.addSolid(collision_primitive_r2)
        return collision_node

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name='xarm_lite6', enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name)
        this_dir, this_filename = os.path.split(__file__)
        # left side
        self.body = jl.JLChain(pos=pos, rotmat=rotmat, homeconf=np.zeros(0), name='lft_body_jl')
        self.body.lnks[0]['name'] = "frame"
        self.body.lnks[0]['pos'] = np.array([-.505, -.132, 0])
        self.body.lnks[0]['loc_rotmat'] = np.array([[0, 0, 1],
                                                    [0, -1, 0],
                                                    [1, 0, 0]])
        self.body.lnks[0]['rgba'] = [.55, .55, .55, 1.0]
        self.body.lnks[0]['collision_model'] = cm.CollisionModel(
            os.path.join(this_dir, "meshes", "table.stl"),
            cdprim_type="user_defined", ex_radius=.005,
            userdef_cdprim_fn=self._table_cdnp)
        self.body.finalize()

        self.arm = Nova2(pos=self.body.jnts[-1]['gl_posq'],
                         rotmat=self.body.jnts[-1]['gl_rotmatq'],
                         enable_cc=False)
        self.hnd = Nova2HuriGripper(pos=self.arm.jnts[-1]['gl_posq'],
                                    rotmat=self.arm.jnts[-1]['gl_rotmatq'],
                                    enable_cc=False)

        # tool center point
        self.arm.jlc.flange_jnt_id = -1
        self.arm.jlc._loc_flange_pos = self.hnd.jaw_center_pos
        self.arm.jlc._loc_flange_rotmat = self.hnd.loc_acting_center_rotmat
        # a list of detailed information about objects in hand, see CollisionChecker.add_objinhnd
        self.oih_infos = []
        # collision detection
        if enable_cc:
            self.enable_cc()
        # component map
        self.manipulator_dict['arm'] = self.arm
        self.manipulator_dict['hnd'] = self.arm  # specify which hand is a gripper installed to
        self.hnd_dict['hnd'] = self.hnd
        self.hnd_dict['arm'] = self.hnd

    def enable_cc(self):
        super().enable_cc()
        # raise NotImplementedError
        self.cc.add_cdlnks(self.body, [0])
        self.cc.add_cdlnks(self.arm, [0, 1, 2, 3, 4, 5, 6])
        self.cc.add_cdlnks(self.hnd.body, [0])
        self.cc.add_cdlnks(self.hnd.lft, [1, 2])
        self.cc.add_cdlnks(self.hnd.rgt, [1, 2])
        # lnks used for cd with external stationary objects
        activelist = [self.arm.lnks[2],
                      self.arm.lnks[3],
                      self.arm.lnks[4],
                      self.arm.lnks[5],
                      self.arm.lnks[6],
                      self.hnd.body.lnks[0],
                      self.hnd.lft.lnks[1],
                      self.hnd.lft.lnks[2],
                      self.hnd.rgt.lnks[1],
                      self.hnd.rgt.lnks[2], ]
        self.cc.set_active_cdlnks(activelist)
        # lnks used for arm-body collision detection
        # fromlist = [self.body.lnks[0],
        #             self.arm.lnks[1], ]
        fromlist = [self.body.lnks[0],
                    self.arm.lnks[0],
                    self.arm.lnks[1], ]
        intolist = [self.arm.lnks[3],
                    self.arm.lnks[4],
                    self.arm.lnks[5],
                    self.arm.lnks[6],
                    self.hnd.body.lnks[0],
                    self.hnd.lft.lnks[1],
                    self.hnd.lft.lnks[2],
                    self.hnd.rgt.lnks[1],
                    self.hnd.rgt.lnks[2], ]
        self.cc.set_cdpair(fromlist, intolist)
        # lnks used for arm-body collision detection -- extra
        # fromlist = [self.body.lnks[0]]  # body
        # intolist = [self.arm.lnks[2], ]
        # self.cc.set_cdpair(fromlist, intolist)
        # lnks used for in-arm collision detection
        fromlist = [self.body.lnks[0],
                    self.arm.lnks[2], ]
        intolist = [self.arm.lnks[4],
                    self.arm.lnks[5],
                    self.hnd.body.lnks[0],
                    self.hnd.lft.lnks[1],
                    self.hnd.lft.lnks[2],
                    self.hnd.rgt.lnks[1],
                    self.hnd.rgt.lnks[2]]
        self.cc.set_cdpair(fromlist, intolist)

        fromlist = [self.body.lnks[0],
                    self.arm.lnks[3]]
        intolist = [self.arm.lnks[6],
                    self.hnd.body.lnks[0],
                    self.hnd.lft.lnks[1],
                    self.hnd.lft.lnks[2],
                    self.hnd.rgt.lnks[1],
                    self.hnd.rgt.lnks[2]]
        self.cc.set_cdpair(fromlist, intolist)

    def fix_to(self, pos, rotmat):
        self.pos = pos
        self.rotmat = rotmat
        self.body.fix_to(self.pos, self.rotmat)
        self.arm.fix_to(pos=self.body.jnts[-1]['gl_posq'], rotmat=self.body.jnts[-1]['gl_rotmatq'])
        self.hnd.fix_to(pos=self.arm.jnts[-1]['gl_posq'],
                        rotmat=self.arm.jnts[-1]['gl_rotmatq'])
        # update objects in hand if available
        for obj_info in self.oih_infos:
            gl_pos, gl_rotmat = self.arm.cvt_loc_tcp_to_gl(obj_info['rel_pos'], obj_info['rel_rotmat'])
            obj_info['gl_pos'] = gl_pos
            obj_info['gl_rotmat'] = gl_rotmat

    def fk(self, component_name, jnt_values):
        """
        :param jnt_values: 1x6 or 1x12 nparray
        :hnd_name 'lft_arm', 'rgt_arm', 'both_arm'
        :param component_name:
        :return:
        author: weiwei
        date: 20201208toyonaka
        """

        def update_oih(component_name='arm'):
            # inline function for update objects in hand
            for obj_info in self.oih_infos:
                gl_pos, gl_rotmat = self.cvt_loc_tcp_to_gl(component_name, obj_info['rel_pos'], obj_info['rel_rotmat'])
                obj_info['gl_pos'] = gl_pos
                obj_info['gl_rotmat'] = gl_rotmat

        def update_component(component_name, jnt_values):
            status = self.manipulator_dict[component_name].fk(jnt_values=jnt_values)
            self.hnd_dict[component_name].fix_to(
                pos=self.manipulator_dict[component_name].jnts[-1]['gl_posq'],
                rotmat=self.manipulator_dict[component_name].jnts[-1]['gl_rotmatq'])
            update_oih(component_name=component_name)
            return status

        super().fk(component_name, jnt_values)
        # examine length
        if component_name == 'arm':
            if not isinstance(jnt_values, np.ndarray) or jnt_values.size != self.arm.n_dof:
                raise ValueError(f"An 1x{self.arm.n_dof} npdarray must be specified to move a single arm!")
            return update_component(component_name, jnt_values)
        else:
            raise ValueError("The given component name is not available!")

    def rand_conf(self, component_name):
        """
        override robot_interface.rand_conf
        :param component_name:
        :return:
        author: weiwei
        date: 20210406
        """
        if component_name == 'arm':
            return super().rand_conf(component_name)
        else:
            raise NotImplementedError

    def gen_stickmodel(self,
                       tcp_jnt_id=None,
                       tcp_loc_pos=None,
                       tcp_loc_rotmat=None,
                       toggle_tcp_frame=False,
                       toggle_jnt_frames=False,
                       toggle_flange_frame=False,
                       name='nova2_stickmodel'):
        stickmodel = mc.ModelCollection(name=name)
        self.body.gen_stickmodel(tcp_loc_pos=None,
                                 tcp_loc_rotmat=None,
                                 toggle_tcpcs=False,
                                 toggle_jntscs=toggle_jnt_frames).attach_to(stickmodel)
        self.arm.gen_stickmodel(tcp_jnt_id=tcp_jnt_id,
                                tcp_loc_pos=tcp_loc_pos,
                                tcp_loc_rotmat=tcp_loc_rotmat,
                                toggle_tcpcs=toggle_tcp_frame,
                                toggle_jntscs=toggle_jnt_frames,
                                toggle_connjnt=toggle_flange_frame).attach_to(stickmodel)
        self.hnd.gen_stickmodel(toggle_tcp_frame=False, toggle_jnt_frames=toggle_jnt_frames).attach_to(stickmodel)
        return stickmodel

    def gen_meshmodel(self,
                      tcp_jnt_id=None,
                      tcp_loc_pos=None,
                      tcp_loc_rotmat=None,
                      toggle_tcp_frame=False,
                      toggle_jnt_frames=False,
                      rgba=None,
                      name='nova2_meshmodel'):
        mm_collection = mc.ModelCollection(name=name)
        self.body.gen_meshmodel(tcp_loc_pos=None,
                                tcp_loc_rotmat=None,
                                toggle_tcpcs=False,
                                toggle_jntscs=toggle_jnt_frames,
                                rgba=rgba).attach_to(mm_collection)
        self.arm.gen_meshmodel(toggle_tcp_frame=toggle_tcp_frame, toggle_jnt_frames=toggle_jnt_frames,
                               rgba=rgba).attach_to(mm_collection)
        self.hnd.gen_meshmodel(toggle_tcp_frame=False,
                               toggle_jnt_frames=toggle_jnt_frames,
                               rgba=rgba).attach_to(mm_collection)
        for obj_info in self.oih_infos:
            objcm = obj_info['collision_model']
            objcm.set_pos(obj_info['gl_pos'])
            objcm.set_rotmat(obj_info['gl_rotmat'])
            objcm.copy().attach_to(mm_collection)
        return mm_collection

    # def is_collided(self, obstacle_list=None, other_robot_list=None, toggle_contacts=False):
    #     """
    #     Interface for "is cdprimit collided", must be implemented in child class
    #     :param obstacle_list:
    #     :param other_robot_list:
    #     :param toggle_contacts: debug
    #     :return: see CollisionChecker is_collided for details
    #     author: weiwei
    #     date: 20201223
    #     """
    #     is_collided = super().is_collided(obstacle_list, other_robot_list, toggle_contacts)
    #     # if len(self.oih_infos) < 1:
    #     #     return is_collided
    #     tcp_constraints = False
    #     tcp_rot = self.get_gl_tcp('arm')[1]
    #     print(tcp_rot[:3, 2].dot(np.array([-1, 0, 0])))
    #     if tcp_rot[:3, 2].dot(np.array([-1, 0, 0])) < .85:
    #         # print(tcp_rot[:3, 2])
    #         # print(tcp_rot[:3, 2].dot(np.array([0, 0, 1])))
    #         tcp_constraints = True
    #     if not toggle_contacts:
    #         return is_collided or tcp_constraints
    #     else:
    #         return (is_collided[0] or tcp_constraints, is_collided[1])


if __name__ == '__main__':
    import visualization.panda.world as wd
    import modeling.geometric_model as gm

    base = wd.World(cam_pos=[1.46567, -1.81357, 5.0505], lookat_pos=[-.5, 0, 0], up=[1, 0, 0])
    gm.gen_frame().attach_to(base)
    xarm = Nova2WRS(enable_cc=True)
    rand_conf = xarm.rand_conf(component_name='arm')
    xarm.fk('arm', rand_conf)
    xarm_meshmodel = xarm.gen_meshmodel(toggle_tcp_frame=False)
    xarm_meshmodel.attach_to(base)
    xarm_meshmodel.show_cdprimit()
    gm.gen_frame(length=.2).attach_to(base)
    # xarm.gen_stickmodel().attach_to(base)
    print("Is self collided?", xarm.is_collided())
    base.run()

import os
import math
import numpy as np
from panda3d.core import CollisionNode, CollisionBox, Point3
import wrs.basis.robot_math as rm
import wrs.robot_sim.manipulators.sia5.sia5 as sia
import wrs.robot_sim.robots.robot_interface as ri
import wrs.modeling.collision_model as mcm
import wrs.modeling.model_collection as mmc
import wrs.robot_sim._kinematics.jlchain as rkjlc


class SDA5F(ri.RobotInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name='ur3dual', enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name)
        this_dir, this_filename = os.path.split(__file__)
        # left side
        self.lft_body = rkjlc.JLChain(pos=pos, rotmat=rotmat, n_dof=1, name='lft_body_jl')
        self.lft_body.jnts[1]['loc_pos'] = np.array([0.045, 0, 0.7296])
        self.lft_body.jnts[2]['loc_pos'] = np.array([0.15, 0.101, 0.1704])
        self.lft_body.jnts[2]['gl_rotmat'] = rm.rotmat_from_euler(-math.pi / 2.0, -math.pi / 2.0, 0)
        self.lft_body.lnks[0]['name'] = "sda5f_lft_body"
        self.lft_body.lnks[0]['loc_pos'] = np.array([0, 0, 0])
        self.lft_body.lnks[0]['collision_model'] = mcm.CollisionModel(
            os.path.join(this_dir, "meshes", "base_link.stl"),
            cdprim_type="user_defined", ex_radius=.005,
            userdef_cdprim_fn=self._base_combined_cdnp)
        self.lft_body.lnks[0]['rgba'] = [.7, .7, .7, 1.0]
        self.lft_body.lnks[1]['name'] = "sda5f_lft_torso"
        self.lft_body.lnks[1]['loc_pos'] = np.array([0, 0, 0])
        self.lft_body.lnks[1]['collision_model'] = mcm.CollisionModel(
            os.path.join(this_dir, "meshes", "torso_link.stl"),
            cdprim_type="user_defined", ex_radius=.005,
            userdef_cdprim_fn=self._torso_combined_cdnp)
        self.lft_body.lnks[1]['rgba'] = [.7, .7, .7, 1.0]
        self.lft_body.finalize()
        lft_arm_homeconf = np.zeros(7)
        # lft_arm_homeconf[0] = math.pi / 3.0
        # lft_arm_homeconf[1] = -math.pi * 1.0 / 3.0
        # lft_arm_homeconf[2] = -math.pi * 2.0 / 3.0
        # lft_arm_homeconf[3] = math.pi
        # lft_arm_homeconf[4] = -math.pi / 2.0
        self.lft_arm = sia.SIA5(pos=self.lft_body.jnts[-1]['gl_posq'],
                                rotmat=self.lft_body.jnts[-1]['gl_rotmatq'],
                                homeconf=lft_arm_homeconf,
                                enable_cc=False)
        # lft hand offset (if needed)
        self.lft_hnd_offset = np.zeros(3)
        lft_hnd_pos, lft_hnd_rotmat = self.lft_arm.cvt_loc_tcp_to_gl(loc_pos=self.lft_hnd_offset)
        self.lft_hnd = rtq.Robotiq85(pos=lft_hnd_pos,
                                     rotmat=self.lft_arm.jnts[-1]['gl_rotmatq'],
                                     enable_cc=False)
        # right side
        self.rgt_body = jl.JLChain(pos=pos, rotmat=rotmat, home_conf=np.zeros(1), name='rgt_body_jl')
        self.rgt_body.jnts[1]['loc_pos'] = np.array([0.045, 0, 0.7296])  # right from robot_s view
        self.rgt_body.jnts[2]['loc_pos'] = np.array([0.15, -0.101, 0.1704])
        self.rgt_body.jnts[2]['gl_rotmat'] = rm.rotmat_from_euler(math.pi / 2.0, -math.pi / 2.0, 0)
        self.rgt_body.lnks[0]['name'] = "sda5f_rgt_body"
        self.rgt_body.lnks[0]['loc_pos'] = np.array([0, 0, 0])
        self.rgt_body.lnks[0]['mesh_file'] = None
        self.rgt_body.lnks[1]['name'] = "sda5f_rgt_torso"
        self.rgt_body.lnks[1]['loc_pos'] = np.array([0, 0, 0])
        self.rgt_body.lnks[1]['mesh_file'] = None
        self.rgt_body.finalize()
        rgt_arm_homeconf = np.zeros(7)
        # rgt_arm_homeconf[0] = -math.pi * 1.0 / 3.0
        # rgt_arm_homeconf[1] = -math.pi * 2.0 / 3.0
        # rgt_arm_homeconf[2] = math.pi * 2.0 / 3.0
        # rgt_arm_homeconf[4] = math.pi / 2.0
        self.rgt_arm = sia.SIA5(pos=self.rgt_body.jnts[-1]['gl_posq'],
                                rotmat=self.rgt_body.jnts[-1]['gl_rotmatq'],
                                homeconf=rgt_arm_homeconf,
                                enable_cc=False)
        # rgt hand offset (if needed)
        self.rgt_hnd_offset = np.zeros(3)
        rgt_hnd_pos, rgt_hnd_rotmat = self.rgt_arm.cvt_loc_tcp_to_gl(loc_pos=self.rgt_hnd_offset)
        # TODO replace using copy
        self.rgt_hnd = rtq.Robotiq85(pos=rgt_hnd_pos,
                                     rotmat=self.rgt_arm.jnts[-1]['gl_rotmatq'],
                                     enable_cc=False)
        # tool center point
        # lft
        self.lft_arm.tcp_jnt_id = -1
        self.lft_arm.loc_tcp_pos = np.array([0, 0, .145])
        self.lft_arm.loc_tcp_rotmat = np.eye(3)
        # rgt
        self.rgt_arm.tcp_jnt_id = -1
        self.rgt_arm.loc_tcp_pos = np.array([0, 0, .145])
        self.rgt_arm.loc_tcp_rotmat = np.eye(3)
        # a list of detailed information about objects in hand, see CollisionChecker.add_objinhnd
        self.lft_oih_infos = []
        self.rgt_oih_infos = []
        # collision detection
        if enable_cc:
            self.enable_cc()
        self.manipulator_dict['rgt_arm'] = self.rgt_arm
        self.manipulator_dict['lft_arm'] = self.lft_arm
        self.manipulator_dict['rgt_hnd'] = self.rgt_arm  # specify which hand is a grippers installed to
        self.manipulator_dict['lft_hnd'] = self.lft_arm  # specify which hand is a grippers installed to
        self.hnd_dict['rgt_hnd'] = self.rgt_hnd
        self.hnd_dict['lft_hnd'] = self.lft_hnd

    @staticmethod
    def _base_combined_cdnp(name, radius):
        collision_node = CollisionNode(name)
        collision_primitive_c0 = CollisionBox(Point3(.0, 0.0, 0.225),
                                              x=.14 + radius, y=.14 + radius, z=.225 + radius)
        collision_node.addSolid(collision_primitive_c0)
        collision_primitive_c1 = CollisionBox(Point3(0.031, 0.0, 0.73),
                                              x=.0855 + radius, y=.0855 + radius, z=.27 + radius)
        collision_node.addSolid(collision_primitive_c1)
        return collision_node

    @staticmethod
    def _torso_combined_cdnp(name, radius):
        collision_node = CollisionNode(name)
        collision_primitive_c2 = CollisionBox(Point3(0.195, 0.0, 0.1704),
                                              x=.085 + radius, y=.101 + radius, z=.09 + radius)
        collision_node.addSolid(collision_primitive_c2)
        return collision_node

    def enable_cc(self):
        super().enable_cc()
        # raise NotImplementedError

    def move_to(self, pos, rotmat):
        self.pos = pos
        self.rotmat = rotmat
        self.lft_body.fix_to(self.pos, self.rotmat)
        self.lft_arm.fix_to(pos=self.lft_body.jnts[-1]['gl_posq'], rotmat=self.lft_body.jnts[-1]['gl_rotmatq'])
        lft_hnd_pos, lft_hnd_rotmat = self.lft_arm.get_worldpose(relpos=self.rgt_hnd_offset)
        self.lft_hnd.fix_to(pos=lft_hnd_pos, rotmat=lft_hnd_rotmat)
        self.rgt_body.fix_to(self.pos, self.rotmat)
        self.rgt_arm.fix_to(pos=self.rgt_body.jnts[-1]['gl_posq'], rotmat=self.rgt_body.jnts[-1]['gl_rotmatq'])
        rgt_hnd_pos, rgt_hnd_rotmat = self.rgt_arm.get_worldpose(relpos=self.rgt_hnd_offset)
        self.rgt_hnd.fix_to(pos=rgt_hnd_pos, rotmat=rgt_hnd_rotmat)

    def fk(self, component_name, jnt_values):
        """
        :param jnt_values: 1x7 or 1x14 nparray
        :hnd_name 'lft_arm', 'rgt_arm', 'both_arm'
        :param component_name:
        :return:
        author: weiwei
        date: 20201208toyonaka
        """

        def update_oih(component_name='rgt_arm'):
            # inline function for update objects in hand
            if component_name == 'rgt_arm':
                oih_info_list = self.rgt_oih_infos
            elif component_name == 'lft_arm':
                oih_info_list = self.lft_oih_infos
            for obj_info in oih_info_list:
                gl_pos, gl_rotmat = self.cvt_loc_tcp_to_gl(component_name, obj_info['rel_pos'], obj_info['rel_rotmat'])
                obj_info['gl_pos'] = gl_pos
                obj_info['gl_rotmat'] = gl_rotmat

        def update_component(component_name, jnt_values):
            status = self.manipulator_dict[component_name].fk(joint_values=jnt_values)
            self.get_hnd_on_manipulator(component_name).fix_to(
                pos=self.manipulator_dict[component_name].jnts[-1]['gl_posq'],
                rotmat=self.manipulator_dict[component_name].jnts[-1]['gl_rotmatq'])
            update_oih(component_name=component_name)
            return status

        super().fk(component_name, jnt_values)
        # examine axis_length
        if component_name == 'lft_arm' or component_name == 'rgt_arm':
            if not isinstance(jnt_values, np.ndarray) or jnt_values.size != 6:
                raise ValueError("An 1x6 npdarray must be specified to move a single arm!")
            return update_component(component_name, jnt_values)
        elif component_name == 'both_arm':
            if (jnt_values.size != 12):
                raise ValueError("A 1x12 npdarrays must be specified to move both arm!")
            status_lft = update_component('lft_arm', jnt_values[0:6])
            status_rgt = update_component('rgt_arm', jnt_values[6:12])
            return "succ" if status_lft == "succ" and status_rgt == "succ" else "out_of_rng"
        elif component_name == 'all':
            raise NotImplementedError
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
        if component_name == 'lft_arm' or component_name == 'rgt_arm':
            return super().rand_conf(component_name)
        elif component_name == 'both_arm':
            return np.hstack((super().rand_conf('lft_arm'), super().rand_conf('rgt_arm')))
        else:
            raise NotImplementedError

    def gen_stickmodel(self,
                       tcp_jnt_id=None,
                       tcp_loc_pos=None,
                       tcp_loc_rotmat=None,
                       toggle_tcp_frame=False,
                       toggle_jnt_frame=False,
                       toggle_connjnt=False,
                       name='sda5f_stickmodel'):
        stickmodel = mc.ModelCollection(name=name)
        self.lft_body.gen_stickmodel(tcp_loc_pos=None,
                                     tcp_loc_rotmat=None,
                                     toggle_tcp_frame=False,
                                     toggle_jnt_frame=toggle_jnt_frame).attach_to(stickmodel)
        self.lft_arm.gen_stickmodel(tcp_jnt_id=tcp_jnt_id,
                                    tcp_loc_pos=tcp_loc_pos,
                                    tcp_loc_rotmat=tcp_loc_rotmat,
                                    toggle_tcp_frame=toggle_tcp_frame,
                                    toggle_jnt_frame=toggle_jnt_frame,
                                    toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        self.lft_hnd.gen_stickmodel(toggle_tcp_frame=False, toggle_jnt_frames=toggle_jnt_frame).attach_to(stickmodel)
        self.rgt_body.gen_stickmodel(tcp_loc_pos=None,
                                     tcp_loc_rotmat=None,
                                     toggle_tcp_frame=False,
                                     toggle_jnt_frame=toggle_jnt_frame).attach_to(stickmodel)
        self.rgt_arm.gen_stickmodel(tcp_jnt_id=tcp_jnt_id,
                                    tcp_loc_pos=tcp_loc_pos,
                                    tcp_loc_rotmat=tcp_loc_rotmat,
                                    toggle_tcp_frame=toggle_tcp_frame,
                                    toggle_jnt_frame=toggle_jnt_frame,
                                    toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        self.rgt_hnd.gen_stickmodel(toggle_tcp_frame=False, toggle_jnt_frames=toggle_jnt_frame).attach_to(stickmodel)
        return stickmodel

    def gen_meshmodel(self,
                      tcp_jnt_id=None,
                      tcp_loc_pos=None,
                      tcp_loc_rotmat=None,
                      toggle_tcp_frame=False,
                      toggle_jnt_frame=False,
                      rgba=None,
                      name='sda5f_meshmodel'):
        meshmodel = mc.ModelCollection(name=name)
        self.lft_body.gen_mesh_model(tcp_loc_pos=None,
                                     tcp_loc_rotmat=None,
                                     toggle_tcp_frame=False,
                                     toggle_jnt_frame=toggle_jnt_frame,
                                     rgba=rgba).attach_to(meshmodel)
        self.lft_arm.gen_meshmodel(toggle_tcp_frame=toggle_tcp_frame, toggle_jnt_frames=toggle_jnt_frame,
                                   rgba=rgba).attach_to(meshmodel)
        self.lft_hnd.gen_meshmodel(toggle_tcp_frame=False,
                                   toggle_jnt_frames=toggle_jnt_frame,
                                   rgba=rgba).attach_to(meshmodel)
        self.rgt_arm.gen_meshmodel(toggle_tcp_frame=toggle_tcp_frame, toggle_jnt_frames=toggle_jnt_frame,
                                   rgba=rgba).attach_to(meshmodel)
        self.rgt_hnd.gen_meshmodel(toggle_tcp_frame=False,
                                   toggle_jnt_frames=toggle_jnt_frame,
                                   rgba=rgba).attach_to(meshmodel)
        for obj_info in self.lft_oih_infos:
            objcm = obj_info['collision_model']
            objcm.set_pos(obj_info['gl_pos'])
            objcm.set_rotmat(obj_info['gl_rotmat'])
            objcm.copy().attach_to(meshmodel)
        for obj_info in self.rgt_oih_infos:
            objcm = obj_info['collision_model']
            objcm.set_pos(obj_info['gl_pos'])
            objcm.set_rotmat(obj_info['gl_rotmat'])
            objcm.copy().attach_to(meshmodel)
        return meshmodel


if __name__ == '__main__':
    import wrs.visualization.panda.world as wd

    base = wd.World(cam_pos=[3, 0, 3], lookat_pos=[0, 0, 1])
    gm.gen_frame().attach_to(base)
    sdarbt = SDA5F()
    sdarbt_meshmodel = sdarbt.gen_meshmodel()
    sdarbt_meshmodel.attach_to(base)
    # sdarbt_meshmodel.show_cdprimit()
    sdarbt.gen_stickmodel().attach_to(base)
    base.run()

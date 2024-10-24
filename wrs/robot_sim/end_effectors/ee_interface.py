import numpy as np

import wrs.basis.robot_math as rm
import wrs.modeling.geometric_model as mgm
import wrs.modeling.collision_model as mcm
import wrs.robot_sim._kinematics.jl as rkjl
import wrs.robot_sim._kinematics.model_generator as rkmg


class EEInterface(object):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), cdmesh_type=mcm.const.CDMeshType.AABB, name="end_effector"):
        self.name = name
        self._pos = pos
        self._rotmat = rotmat
        self.cdmesh_type = cdmesh_type
        # joints
        # -- coupling --
        # no coupling by default, change the pos if the coupling existed
        # use loc flange create non-straight couplings
        self.coupling = rkjl.Anchor(name=name + "_coupling", pos=self._pos, rotmat=self._rotmat, n_flange=1)
        # acting center of the tool
        self.loc_acting_center_pos = np.zeros(3)
        self.loc_acting_center_rotmat = np.eye(3)
        # cd mesh collection for precise collision checking
        self.cdmesh_elements = []
        # object grasped/held/attached to end-effector; oiee = object in end-effector
        self.oiee_list = []
        self.oiee_list_bk = []
        self.oiee_pose_list_bk = []

    @staticmethod
    def assert_oiee_decorator(method):
        def wrapper(self, *args, **kwargs):
            if len(self.oiee_list) > 0:
                raise ValueError("The hand is holding objects!")
            else:
                return method(self, *args, **kwargs)

        return wrapper

    @property
    def pos(self):
        return self._pos

    @property
    def rotmat(self):
        return self._rotmat

    @property
    def gl_acting_center_pose(self):
        gl_acting_center_pos = self._rotmat @ self.loc_acting_center_pos + self._pos
        gl_acting_center_rotmat = self._rotmat @ self.loc_acting_center_rotmat
        return (gl_acting_center_pos, gl_acting_center_rotmat)

    def update_oiee(self):
        """
        :return:
        author: weiwei
        date: 20230807
        """
        for oiee in self.oiee_list:
            # oiee.install_onto(pos=self._pos, rotmat=self._rotmat)
            oiee.root_pose = (self._pos, self._rotmat)

    @assert_oiee_decorator
    def hold(self, obj_cmodel, **kwargs):
        """
        the obj_cmodel is saved into an oiee_list, while considering its relative pose to the ee's pos and rotmat
        **kwargs is for polyphorism purpose
        :param obj_cmodel: a collision model
        :return:
        author: weiwei
        date: 20230811
        """
        loc_pos, loc_rotmat = rm.rel_pose(self._pos, self._rotmat, obj_cmodel.pos, obj_cmodel.rotmat)
        obj_lnk = rkjl.Link(loc_pos=loc_pos, loc_rotmat=loc_rotmat, cmodel=obj_cmodel)
        self.oiee_list.append(obj_lnk)
        self.update_oiee()
        return obj_lnk

    def release(self, obj_cmodel, **kwargs):
        """
        the obj_cmodel is saved into an oiee_list, while considering its relative pose to the ee's pos and rotmat
        **kwargs is for polyphorism purpose
        :param obj_cmodel: a collision model
        :return:
        author: weiwei
        date: 20240228
        """
        is_found = False
        for obj_lnk in self.oiee_list:
            if obj_lnk.cmodel is obj_cmodel:
                is_found = True
                self.oiee_list.remove(obj_lnk)
                break
        if not is_found:
            raise ValueError("The specified object is not held in the hand!")
        return obj_lnk

    def release_all(self):
        """
        release all objects held in ee
        :return:
        author: weiwei
        date: 20240314
        """
        self.oiee_list = []


    def backup_state(self):
        self.oiee_list_bk.append(self.oiee_list.copy())
        self.oiee_pose_list_bk.append([oiee.loc_pose for oiee in self.oiee_list])

    def restore_state(self):
        self.oiee_list = self.oiee_list_bk.pop()
        oiee_pose_list = self.oiee_pose_list_bk.pop()
        for i, oiee in enumerate(self.oiee_list):
            oiee.loc_pose = oiee_pose_list[i]
        self.update_oiee()

    def change_ee_values(self, ee_values):
        raise NotImplementedError

    def get_ee_values(self):
        return None

    def is_mesh_collided(self, cmodel_list=None, toggle_dbg=False):
        """
        check collision of cd meshes
        :param cmodel_list:
        :param toggle_dbg: show cd mesh and draw colliding points in case of collision
        :return:
        """
        if cmodel_list is None:
            return False
        if isinstance(cmodel_list, mcm.CollisionModel):
            cmodel_list = [cmodel_list]
        for i, cdme in enumerate(self.cdmesh_elements):
            if cdme.cmodel is not None:
                is_collided, collision_points = cdme.cmodel.is_mcdwith(cmodel_list, True)
                if is_collided:
                    if toggle_dbg:
                        mgm.GeometricModel(cdme.cmodel).attach_to(base)
                        print(collision_points)
                        for point in collision_points:
                            mgm.gen_sphere(point, radius=.01).attach_to(base)
                        print("mesh collided")
                    return True
        return False

    def fix_to(self, pos, rotmat):
        raise NotImplementedError

    @assert_oiee_decorator
    def align_acting_center_by_twovecs(self,
                                       acting_center_pos,
                                       approaching_vec,
                                       side_vec):
        """
        align acting center to a frame decided by two vectors
        :param acting_center_pos:
        :param approaching_vec:
        :param side_vec:
        :return:
        """
        acting_center_rotmat = np.eye(3)
        acting_center_rotmat[:, 2] = rm.unit_vector(approaching_vec)
        acting_center_rotmat[:, 1] = rm.unit_vector(side_vec)
        acting_center_rotmat[:, 0] = np.cross(acting_center_rotmat[:3, 1], acting_center_rotmat[:3, 2])
        return self.align_acting_center_by_pose(acting_center_pos=acting_center_pos,
                                                acting_center_rotmat=acting_center_rotmat)

    @assert_oiee_decorator
    def align_acting_center_by_pose(self, acting_center_pos, acting_center_rotmat):
        """
        align acting center to a frame decided by pos and rotmat
        :param acting_center_pos:
        :param acting_center_rotmat:
        :return:
        """
        ee_root_rotmat = acting_center_rotmat.dot(self.loc_acting_center_rotmat.T)
        ee_root_pos = acting_center_pos - ee_root_rotmat.dot(self.loc_acting_center_pos)
        self.fix_to(ee_root_pos, ee_root_rotmat)
        return [acting_center_pos, acting_center_rotmat, ee_root_pos, ee_root_rotmat]

    def gen_stickmodel(self, toggle_tcp_frame=False, toggle_jnt_frames=False, name='ee_stickmodel'):
        raise NotImplementedError

    def gen_meshmodel(self,
                      rgb=None,
                      alpha=None,
                      toggle_tcp_frame=False,
                      toggle_jnt_frames=False,
                      toggle_cdprim=False,
                      toggle_cdmesh=False,
                      name='ee_meshmodel'):
        raise NotImplementedError

    def _gen_oiee_meshmodel(self,
                            m_col,
                            rgb=None,
                            alpha=None,
                            toggle_cdprim=False,
                            toggle_cdmesh=False,
                            toggle_frame=False):
        """
        :return:
        author: weiwei
        date: 20230807
        """
        for oiee in self.oiee_list:
            if toggle_frame:
                oiee.cmodel.show_local_frame()
            rkmg.gen_lnk_mesh(lnk=oiee, rgb=rgb, alpha=alpha, toggle_cdprim=toggle_cdprim,
                              toggle_cdmesh=toggle_cdmesh).attach_to(m_col)

    def _toggle_tcp_frame(self, m_col):
        gl_acting_center_pos = self._rotmat @ self.loc_acting_center_pos + self._pos
        gl_acting_center_rotmat = self._rotmat @ self.loc_acting_center_rotmat
        print(gl_acting_center_rotmat)
        rkmg.gen_indicated_frame(spos=self._pos,
                                 gl_pos=gl_acting_center_pos,
                                 gl_rotmat=gl_acting_center_rotmat).attach_to(m_col)

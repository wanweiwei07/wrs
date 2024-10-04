import numpy as np
from wrs import basis as rm, modeling as cm


class Base(cm.CollisionModel):
    def __init__(self, file):
        super().__init__(initor=file, ex_radius=.009)
        self._hole_pos_list = []
        self._pos_z0 = .042
        self._pos_y0 = -.0315
        self._pos_x0 = .0495
        self._y_step = .009
        self._x_step = .009
        self._y_holes = 8
        self._x_holes = 12
        self._reinitialize()

    def _reinitialize(self):
        self._hole_pos_list_0 = []
        pos_z = self._pos_z0
        for id_y in range(self._y_holes):
            pos_y = self._pos_y0 + self._y_step * id_y
            if id_y % 2 == 0:
                for id_x in range(self._x_holes):
                    pos_x = self._pos_x0 - self._x_step * id_x
                    self._hole_pos_list_0.append(np.array([pos_y, pos_x, pos_z]))
            else:
                for id_x in range(self._x_holes):
                    pos_x = self._x_step * id_x - self._pos_x0
                    self._hole_pos_list_0.append(np.array([pos_y, pos_x, pos_z]))
        self._hole_pos_list = self._hole_pos_list_0.reverse()

    def _update_hole_pos_list(self):
        self._hole_pos_list = list(self.get_rotmat().dot(np.asarray(self._hole_pos_list_0).T).T + self.get_pos())

    def set_pos(self, pos):
        super().set_pos(pos)
        self._update_hole_pos_list()

    def set_rotmat(self, rotmat):
        super().set_rotmat(rotmat)
        self._update_hole_pos_list()

    def set_pose(self, pos, rotmat):
        super().set_pose(pos, rotmat)
        self._update_hole_pos_list()

    def set_homomat(self, npmat4):
        super().set_homomat(npmat4)
        self._update_hole_pos_list()

    def set_pose_from_2d_points(self, point1, point2, point3):
        """
        1
        \
        \
        \
        2----3
        """
        x1, y1 = point1
        x2, y2 = point2
        x3, y3 = point3
        pos = np.array([(x1 + x3) / 2, (y1 + y3) / 2, 0])
        super().set_pos(pos)
        rot_angle = rm.angle_between_2d_vecs((x2 - x1, y2 - y1), (0, 1))
        rotmat = rm.rotmat_from_axangle(np.array([0, 0, 1]), rot_angle)
        super().set_rotmat(rotmat)
        self._update_hole_pos_list()

    def get_rack_hole_pose(self, id_x, id_y):
        """
        get the rack hole pose given the hole id
        :param id_x, id_y: (0,0) indicates the upper_left corner when the rack is at a 12 row x 8 column view
        :return:
        author: weiwei
        date: 20220403
        """
        id = id_y * 8 + id_x
        return self._hole_pos_list[id], self.get_rotmat()

    def copy(self):
        return Base(self)


class Base96(Base):
    def __init__(self, file):
        super().__init__(file)


class Microplate96(Base96):
    def __init__(self, file):
        super().__init__(file)


class Base6(Base):
    def __init__(self, file):
        super().__init__(file)
        self._pos_z0 = .02
        self._pos_y0 = -.0295
        self._pos_x0 = .049
        self._y_step = .039
        self._x_step = .039
        self._y_holes = 2
        self._x_holes = 3
        self._reinitialize()

    def copy(self):
        return Base6(self)

    def _reinitialize(self):
        self._hole_pos_list_0 = []
        pos_z = self._pos_z0
        for id_y in range(self._y_holes):
            pos_y = self._pos_y0 + self._y_step * id_y
            for id_x in range(self._x_holes):
                pos_x = self._pos_x0 - self._x_step * id_x
                self._hole_pos_list_0.append(np.array([pos_y, pos_x, pos_z]))
        self._hole_pos_list = self._hole_pos_list_0.reverse()


class Dish6(Base6):
    def __init__(self, file):
        super().__init__(file)


def search_reachable_configuration(rbt_s,
                                   ee_s,
                                   component_name,
                                   tgt_pos,
                                   cone_axis,
                                   cone_angle=0,
                                   rotation_interval=np.radians(22.5),
                                   obstacle_list=[],
                                   seed_jnt_values=None,
                                   toggle_debug=False) -> np.typing.NDArray:
    """
    search reachable configuration in a cone
    when the cone_angle is 0, the function degenerates into a search around the cone_axis
    :param rbt_s: instance of a robot
    :param ee_s: instance of an end_type-effector
    :param tgt_pos:
    :param cone_axis:
    :param cone_angle:
    :param granularity:
    :param obstacle_list
    :return:
    author: weiwei
    date: 20220404
    """
    jnt_values_bk = rbt_s.get_jnt_values(component_name=component_name)
    if seed_jnt_values is None:
        seed_jnt_values = jnt_values_bk
    rotmat_list = []
    if cone_angle != 0:
        rotmat_list = rm.gen_icorotmats(icolevel=3,
                                        rotation_interval=rotation_interval,
                                        crop_normal=-cone_axis,
                                        crop_angle=cone_angle,
                                        toggle_flat=True)
    else:
        rotmat = rm.rotmat_from_axangle([0, 0, 1], 0).dot(rm.rotmat_from_normal(cone_axis))
        for angle in np.linspace(0, np.pi * 2, int(np.pi * 2 / rotation_interval), endpoint=False):
            rotmat_list.append(rm.rotmat_from_axangle([0, 0, 1], -angle).dot(rotmat))
    print("======new search!")
    for i, rotmat in enumerate(rotmat_list):
        jnt_values = rbt_s.ik(component_name=component_name,
                              tgt_pos=tgt_pos,
                              tgt_rotmat=rotmat,
                              seed_jnt_values=seed_jnt_values)
        if jnt_values is not None:
            rbt_s.fk(joint_values=jnt_values)
            if rbt_s.is_collided(obstacle_list=obstacle_list):
                if toggle_debug:
                    rbt_s.gen_mesh_model(rgba=[.9, .5, 0, .3]).attach_to(base)
            else:
                if toggle_debug:
                    rbt_s.gen_mesh_model().attach_to(base)
                if not toggle_debug:
                    rbt_s.fk(component_name=component_name,
                             joint_values=jnt_values_bk)
                    print("times tried ", i)
                    return jnt_values
        else:
            if toggle_debug:
                ee_s.grip_at_by_pose(jaw_center_pos=tgt_pos,
                                     jaw_center_rotmat=rotmat,
                                     jaw_width=0)
                ee_s.gen_mesh_model(rgba=[1, 0, 0, .3]).attach_to(base)
    rbt_s.fk(component_name=component_name, joint_values=jnt_values_bk)
    return None

import scipy.optimize as sopt
import pickle
import wrs.basis.robot_math as rm


def load_calibration_data(file="./depth_sensor_calib_mat.pkl",
                          has_sensor_and_real_points=False):
    """
    :param file:
    :param has_sensor_and_real_points:
    :return:
    author: weiwei
    date: 20210519
    """
    if has_sensor_and_real_points:
        affine_mat, pos_in_real_array, pos_in_sensor_array = pickle.load(open(file, "rb"))
    else:
        affine_mat = pickle.load(open(file, "rb"))
        pos_in_real_array = None
        pos_in_sensor_array = None
    return affine_mat, pos_in_real_array, pos_in_sensor_array


class DepthCaliberator(object):

    def __init__(self, robot_x, robot_s):
        self.robot_x = robot_x
        self.robot_s = robot_s

    def _find_tcp_in_sensor(self, component_name, action_pos, action_rotmat, sensor_marker_handler):
        """
        find the robot_s tcp's pos and rotmat in the sensor coordinate system
        :param component_name:
        :param loc_acting_center_pos, action_rotmat:
        :param marker_callback:
        :return: [estiamted tcp center in sensor, major_radius of the sphere formed by markers]
        author: weiwei
        date: 20210408
        """

        def _fit_sphere(p, coords):
            x0, y0, z0, radius = p
            x, y, z = coords.T
            return rm.np.sqrt((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2)

        _err_fit_sphere = lambda p, x: _fit_sphere(p, x) - p[3]

        marker_pos_in_sensor_list = []
        rot_range_x = [rm.np.array([1, 0, 0]), [-30, -15, 0, 15, 30]]
        rot_range_y = [rm.np.array([0, 1, 0]), [-30, -15, 15, 30]]
        rot_range_z = [rm.np.array([0, 0, 1]), [-90, -60, -30, 30, 60]]
        range_axes = [rot_range_x, rot_range_y, rot_range_z]
        last_jnt_values = self.robot_x.lft_arm_hnd.get_jnt_values()
        jnt_values_bk = self.robot_s.get_jnt_values(component_name)
        for ax_id in range(3):
            axis = range_axes[ax_id][0]
            for angle in range_axes[ax_id][1]:
                goal_pos = action_pos
                goal_rotmat = rm.np.dot(rm.rotmat_from_axangle(axis, angle), action_rotmat)
                jnt_values = self.robot_s.ik(component_name=component_name,
                                             tgt_pos=goal_pos,
                                             tgt_rotmat=goal_rotmat,
                                             seed_jnt_values=last_jnt_values)
                self.robot_s.fk(component_name=component_name, joint_values=jnt_values)
                if jnt_values is not None and not self.robot_s.is_collided():
                    last_jnt_values = jnt_values
                    self.robot_x.move_jnts(component_name, jnt_values)
                    marker_pos_in_sensor = sensor_marker_handler.get_marker_center()
                    if marker_pos_in_sensor is not None:
                        marker_pos_in_sensor_list.append(marker_pos_in_sensor)
        self.robot_s.fk(component_name=component_name, joint_values=jnt_values_bk)
        if len(marker_pos_in_sensor_list) < 3:
            return [None, None]
        center_in_camera_coords_array = rm.np.asarray(marker_pos_in_sensor_list)
        # try:
        initial_guess = rm.np.ones(4) * .001
        initial_guess[:3] = rm.np.mean(center_in_camera_coords_array, axis=0)
        final_estimate, flag = sopt.leastsq(_err_fit_sphere, initial_guess, args=(center_in_camera_coords_array,))
        if len(final_estimate) == 0:
            return [None, None]
        return rm.np.array(final_estimate[:3]), final_estimate[3]

    def find_board_center_in_hand(self,
                                  component_name,
                                  sensor_marker_handler,
                                  action_center_pos=rm.np.array([.3, -.05, .2]),
                                  action_center_rotmat=rm.np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]).T,
                                  action_dist=.1):
        """
        :param component_name:
        :param sensor_marker_handler:
        :param action_center_pos:
        :param action_center_rotmat:
        :param action_dist:
        :return:
        author: weiwei
        date: 20210408, 20210519
        """
        tcp_in_sensor, radius_by_markers = self._find_tcp_in_sensor(component_name=component_name,
                                                                    action_pos=action_center_pos,
                                                                    action_rotmat=action_center_rotmat,
                                                                    sensor_marker_handler=sensor_marker_handler)
        jnt_values_bk = self.robot_s.get_jnt_values(component_name)
        # move to action pos, action rotmat
        last_jnt_values = self.robot_x.lft_arm_hnd.get_jnt_values()
        jnt_values = self.robot_s.ik(component_name=component_name,
                                     tgt_pos=action_center_pos,
                                     tgt_rotmat=action_center_rotmat,
                                     seed_jnt_values=last_jnt_values)
        if jnt_values is not None and not self.robot_s.is_collided():
            self.robot_s.fk(component_name=component_name, joint_values=jnt_values)
            last_jnt_values = jnt_values
            self.robot_x.move_jnts(component_name, jnt_values)
            marker_pos_in_sensor = sensor_marker_handler.get_marker_center()
        else:
            raise ValueError("The action center is not reachable. Try a different pos or robtmat!")
        # move to x+action_dist
        action_center_dist_x = action_center_pos + action_center_rotmat[:, 0] * action_dist
        jnt_values = self.robot_s.ik(component_name=component_name,
                                     tgt_pos=action_center_dist_x,
                                     tgt_rotmat=action_center_rotmat,
                                     seed_jnt_values=last_jnt_values)
        if jnt_values is not None and not self.robot_s.is_collided():
            self.robot_s.fk(component_name=component_name, joint_values=jnt_values)
            last_jnt_values = jnt_values
            self.robot_x.move_jnts(component_name, jnt_values)
            marker_pos_xplus_in_sensor = sensor_marker_handler.get_marker_center()
        else:
            raise ValueError("The action center with xplus is not reachable. Try a different pos or robtmat!")
        # move to y+action_dist
        action_center_dist_y = action_center_pos + action_center_rotmat[:, 1] * action_dist
        jnt_values = self.robot_s.ik(component_name=component_name,
                                     tgt_pos=action_center_dist_y,
                                     tgt_rotmat=action_center_rotmat,
                                     seed_jnt_values=last_jnt_values)
        if jnt_values is not None and not self.robot_s.is_collided():
            self.robot_s.fk(component_name=component_name, joint_values=jnt_values)
            last_jnt_values = jnt_values
            self.robot_x.move_jnts(component_name, jnt_values)
            marker_pos_yplus_in_sensor = sensor_marker_handler.get_marker_center()
        else:
            raise ValueError("The action center with yplus is not reachable. Try a different pos or robtmat!")
        # move to z+action_dist
        action_center_dist_z = action_center_pos + action_center_rotmat[:, 2] * action_dist
        jnt_values = self.robot_s.ik(component_name=component_name,
                                     tgt_pos=action_center_dist_z,
                                     tgt_rotmat=action_center_rotmat,
                                     seed_jnt_values=last_jnt_values)
        if jnt_values is not None and not self.robot_s.is_collided():
            self.robot_s.fk(component_name=component_name, joint_values=jnt_values)
            self.robot_x.move_jnts(component_name, jnt_values)
            marker_pos_zplus_in_sensor = sensor_marker_handler.get_marker_center()
        else:
            raise ValueError("The action center with zplus is not reachable. Try a different pos or robtmat!")
        unnormalized_marker_mat_in_sensor = rm.np.array([marker_pos_xplus_in_sensor - marker_pos_in_sensor,
                                                         marker_pos_yplus_in_sensor - marker_pos_in_sensor,
                                                         marker_pos_zplus_in_sensor - marker_pos_in_sensor]).T
        marker_rotmat_in_sensor, r = rm.np.linalg.qr(unnormalized_marker_mat_in_sensor)
        marker_pos_in_hnd = rm.np.dot(marker_rotmat_in_sensor.T, marker_pos_in_sensor - tcp_in_sensor)
        self.robot_s.fk(component_name=component_name, joint_values=jnt_values_bk)
        return marker_pos_in_hnd

    def calibrate(self,
                  component_name,
                  sensor_marker_handler,
                  marker_pos_in_hnd=None,
                  action_pos_list=(rm.vec(.3, -.2, .9), rm.vec(.3, .2, .9),
                                   rm.vec(.4, -.2, .9), rm.vec(.4, .2, .9),
                                   rm.vec(.3, -.2, 1.1), rm.vec(.3, .2, 1.1),
                                   rm.vec(.4, -.2, 1.1), rm.vec(.4, .2, 1.1)),
                  action_rotmat_list=[rm.np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]).T] * 8,
                  save_calib_file='depth_sensor_calib_mat.pkl',
                  save_sensor_and_real_points=False):
        """
        :param marker_pos_in_hnd:
        :param aruco_dict:
        :return:
        author: weiwei
        date: 20191228
        """
        if marker_pos_in_hnd is None:
            marker_pos_in_hnd = self.find_board_center_in_hand(component_name=component_name,
                                                               sensor_marker_handler=sensor_marker_handler)
        pos_in_real_list = []
        pos_in_sensor_list = []
        jnt_values_bk = self.robot_s.get_jnt_values(component_name)
        last_jnt_values = self.robot_x.lft_arm_hnd.get_jnt_values()
        for i, action_pos in enumerate(action_pos_list):
            jnt_values = self.robot_s.ik(component_name=component_name,
                                         tgt_pos=action_pos,
                                         tgt_rotmat=action_rotmat_list[i],
                                         seed_jnt_values=last_jnt_values)
            if jnt_values is not None:
                self.robot_s.fk(component_name=component_name, joint_values=jnt_values)
                last_jnt_values = jnt_values
                if not self.robot_s.is_collided():
                    self.robot_x.move_jnts(component_name, jnt_values)
                    marker_pos_in_sensor = sensor_marker_handler.get_marker_center()
                    if marker_pos_in_sensor is not None:
                        pos_in_real_list.append(action_pos + rm.np.dot(action_rotmat_list[i], marker_pos_in_hnd))
                        pos_in_sensor_list.append(marker_pos_in_sensor)
                else:
                    print(f"The {i}th action pose is collided!")
            else:
                print(f"The {i}th action pose is reachable!")
        self.robot_s.fk(component_name=component_name, joint_values=jnt_values_bk)
        pos_in_real_array = rm.np.array(pos_in_real_list)
        pos_in_sensor_array = rm.np.array(pos_in_sensor_list)
        affine_mat = rm.affine_matrix_from_points(pos_in_sensor_array.T, pos_in_real_array.T)
        if save_sensor_and_real_points:
            data = [affine_mat, pos_in_real_array, pos_in_sensor_array]
        else:
            data = affine_mat
        pickle.dump(data, open('./' + save_calib_file, "wb"))
        return affine_mat

    def refine_with_template(self, affine_mat, template_file):
        """
        refine the affine_mat by matching it with a template
        :param affine_mat:
        :param template_file:
        :return:
        author: weiwei
        date: 20191228, 20210519
        """
        pass

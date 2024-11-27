"""
Author: Hao Chen (chen960216@gmail.com 20221113)
The program to manually calibrate the camera
"""
__VERSION__ = '0.0.1'

import os
from pathlib import Path
import json
from abc import ABC, abstractmethod
import wrs.basis.robot_math as rm
import wrs.modeling.geometric_model as mgm
import wrs.robot_sim.robots.robot_interface as ri


def py2json_data_formatter(data):
    """Format the python data to json format. Only support for np.ndarray, str, int, float ,dict, list"""
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, str) or isinstance(data, float) or isinstance(data, int) or isinstance(data, dict):
        return data
    elif isinstance(data, Path):
        return str(data)
    elif isinstance(data, list):
        return [py2json_data_formatter(d) for d in data]


def dump_json(data, path="", reminder=True):
    path = str(path)
    """Dump the data by json"""
    if reminder and os.path.exists(path):
        option = input(f"File {path} exists. Are you sure to write it, y/n: ")
        print(option)
        option_up = option.upper()
        if option_up == "Y" or option_up == "YES":
            pass
        else:
            return False
    with open(path, "w") as f:
        json.dump(py2json_data_formatter(data), f)
    return True


class ManualCalibrationBase(ABC):
    def __init__(self, rbt_s: ri.RobotInterface, rbt_x, sensor_hdl, init_calib_mat: rm.np.ndarray = None,
                 move_resolution=.001, rotation_resolution=rm.np.radians(5)):
        """
        Class to manually calibrate the point cloud data
        :param rbt_s: The simulation robot
        :param rbt_x: The real robot handler
        :param sensor_hdl: The sensor handler
        :param init_calib_mat: The initial calibration matrix. If it is None, the init calibration matrix will be identity matrix
        :param component_name: component name that mounted the camera
        :param move_resolution: the resolution for manual move adjustment
        :param rotation_resolution: the resolution for manual rotation adjustment
        """
        self._rbt_s = rbt_s
        self._rbt_x = rbt_x
        self._sensor_hdl = sensor_hdl
        self._init_calib_mat = rm.eye(4) if init_calib_mat is None else init_calib_mat

        # variable stores robot plot and the point cloud plot
        self._plot_node_rbt = None
        self._plot_node_pcd = None
        self._pcd = None

        #
        self._key = {}
        self.map_key()
        self.move_resolution = move_resolution
        self.rotation_resolution = rotation_resolution

        # add task
        taskMgr.doMethodLater(.05, self.sync_rbt, "sync rbt", )
        taskMgr.doMethodLater(.02, self.adjust, "manual adjust the mph")
        taskMgr.doMethodLater(.5, self.sync_pcd, "sync mph", )

    @abstractmethod
    def get_pcd(self):
        """
        An abstract method to get the point cloud
        :return: An Nx3 ndarray represents the point cloud
        """
        pass

    @abstractmethod
    def get_rbtx_jnt_values(self):
        """
        An abstract method to get the robot joint angles
        :return: 1xn ndarray, n is degree of freedom of the robot
        """
        pass

    @abstractmethod
    def align_pcd(self, pcd):
        """
        Abstract method to align the mph according to the calibration matrix
        implement the Eye-in-hand or eye-to-hand transformation here
        https://support.zivid.com/en/latest/academy/applications/hand-eye/system-configurations.html
        :return: An Nx3 ndarray represents the aligned point cloud
        """
        pass

    def move_adjust(self, dir, dir_global, key_name=None):
        """
        The abstract method to revise the calibration matrix by moving
        :param dir: The local move motion_vec based on the calibration matrix coordinate
        :param dir_global: The global move motion_vec based on the world coordinate
        :return:
        """
        self._init_calib_mat[:3, 3] = self._init_calib_mat[:3, 3] + dir_global * self.move_resolution

    def rotate_adjust(self, dir, dir_global, key_name=None):
        """
        The abstract method to revise the calibration matrix by rotating
        :param dir: The local motion_vec of the calibration matrix
        :param dir_global: The global motion_vec
        :return:
        """
        self._init_calib_mat[:3, :3] = rm.rotmat_from_axangle(dir_global, rm.radians(
            self.rotation_resolution)) @ self._init_calib_mat[:3, :3]

    def map_key(self, x='w', x_='s', y='a', y_='d', z='q', z_='e', x_cw='z', x_ccw='x', y_cw='c', y_ccw='v', z_cw='b',
                z_ccw='n'):
        def add_key(keys: str or list):
            """
            Add key to  the keymap. The default keymap can be seen in visualization/panda/inputmanager.py
            :param keys: the keys added to the keymap
            """
            assert isinstance(keys, str) or isinstance(keys, list)

            if isinstance(keys, str):
                keys = [keys]

            def set_keys(base, k, v):
                base.inputmgr.keymap[k] = v

            for key in keys:
                if key in base.inputmgr.keymap: continue
                base.inputmgr.keymap[key] = False
                base.inputmgr.accept(key, set_keys, [base, key, True])
                base.inputmgr.accept(key + '-up', set_keys, [base, key, False])

        add_key([x, x_, y, y_, z, z_, x_cw, x_ccw, y_cw, y_ccw, z_cw, z_ccw])
        self._key['x'] = x
        self._key['x_'] = x_
        self._key['y'] = y
        self._key['y_'] = y_
        self._key['z'] = z
        self._key['z_'] = z_
        self._key['x_cw'] = x_cw
        self._key['x_ccw'] = x_ccw
        self._key['y_cw'] = y_cw
        self._key['y_ccw'] = y_ccw
        self._key['z_cw'] = z_cw
        self._key['z_ccw'] = z_ccw

    def sync_pcd(self, task):
        """
        Synchronize the real robot and the simulation robot
        :return: None
        """

        self._pcd = self.get_pcd()
        self.plot()
        return task.again

    def sync_rbt(self, task):
        self._rbt_s.goto_given_conf(self.get_rbtx_jnt_values())
        self.plot()
        return task.again

    def save(self):
        """
        Save manual calibration results
        :return:
        """
        dump_json({'affine_mat': self._init_calib_mat.tolist()}, "manual_calibration.json", reminder=False)

    def plot(self, task=None):
        """
        A task to plot the point cloud and the robot
        :param task:
        :return:
        """
        # clear previous plot
        if self._plot_node_rbt is not None:
            self._plot_node_rbt.detach()
        if self._plot_node_pcd is not None:
            self._plot_node_pcd.detach()
        self._plot_node_rbt = self._rbt_s.gen_meshmodel()
        self._plot_node_rbt.attach_to(base)
        pcd = self._pcd
        if pcd is not None:
            if pcd.shape[1] == 6:
                pcd, pcd_color = pcd[:, :3], pcd[:, 3:6]
                pcd_color_rgba = rm.np.append(pcd_color, rm.np.ones((len(pcd_color), 1)), axis=1)
            else:
                pcd_color_rgba = rm.np.array([1, 1, 1, 1])
            pcd_r = self.align_pcd(pcd)
            self._plot_node_pcd = mgm.gen_pointcloud(points=pcd_r, rgba=pcd_color_rgba)
            mgm.gen_frame(self._init_calib_mat[:3, 3], self._init_calib_mat[:3, :3]).attach_to(self._plot_node_pcd)
            self._plot_node_pcd.attach_to(base)
        if task is not None:
            return task.again

    def adjust(self, task):
        if base.inputmgr.keymap[self._key['x']]:
            self.move_adjust(dir=self._init_calib_mat[:3, 0], dir_global=np.array([1, 0, 0]), key_name='x')
        if base.inputmgr.keymap[self._key['x_']]:
            self.move_adjust(dir=-self._init_calib_mat[:3, 0], dir_global=np.array([-1, 0, 0]), key_name='x_')
        elif base.inputmgr.keymap[self._key['y']]:
            self.move_adjust(dir=self._init_calib_mat[:3, 1], dir_global=np.array([0, 1, 0]), key_name='y')
        elif base.inputmgr.keymap[self._key['y_']]:
            self.move_adjust(dir=-self._init_calib_mat[:3, 1], dir_global=np.array([0, -1, 0]), key_name='y_')
        elif base.inputmgr.keymap[self._key['z']]:
            self.move_adjust(dir=self._init_calib_mat[:3, 2], dir_global=np.array([0, 0, 1]), key_name='z')
        elif base.inputmgr.keymap[self._key['z_']]:
            self.move_adjust(dir=-self._init_calib_mat[:3, 2], dir_global=np.array([0, 0, -1]), key_name='z_')
        elif base.inputmgr.keymap[self._key['x_cw']]:
            self.rotate_adjust(dir=self._init_calib_mat[:3, 0], dir_global=np.array([1, 0, 0]), key_name='x_cw')
        elif base.inputmgr.keymap[self._key['x_ccw']]:
            self.rotate_adjust(dir=-self._init_calib_mat[:3, 0], dir_global=np.array([-1, 0, 0]), key_name='x_ccw')
        elif base.inputmgr.keymap[self._key['y_cw']]:
            self.rotate_adjust(dir=self._init_calib_mat[:3, 1], dir_global=np.array([0, 1, 0]), key_name='y_cw')
        elif base.inputmgr.keymap[self._key['y_ccw']]:
            self.rotate_adjust(dir=-self._init_calib_mat[:3, 1], dir_global=np.array([0, -1, 0]), key_name='y_ccw')
        elif base.inputmgr.keymap[self._key['z_cw']]:
            self.rotate_adjust(dir=self._init_calib_mat[:3, 2], dir_global=np.array([0, 0, 1]), key_name='z_cw')
        elif base.inputmgr.keymap[self._key['z_ccw']]:
            self.rotate_adjust(dir=-self._init_calib_mat[:3, 2], dir_global=np.array([0, 0, -1]), key_name='z_ccw')
        else:
            return task.again
        self.plot()
        self.save()
        return task.again


if __name__ == "__main__":

    class XArmLite6ManualCalib(ManualCalibrationBase):
        """
        Eye in hand example
        """

        def get_pcd(self):
            pcd, pcd_color, _, _ = self._sensor_hdl.get_pcd_texture_depth()
            return rm.np.hstack((pcd, pcd_color))

        def get_rbtx_jnt_values(self):
            return self._rbt_x.get_jnt_values()

        def align_pcd(self, pcd):
            r2cam_mat = self._init_calib_mat
            rbt_pose = self._rbt_x.get_pose()
            w2r_mat = rm.homomat_from_posrot(*rbt_pose)
            w2c_mat = w2r_mat.dot(r2cam_mat)
            return rm.transform_points_by_homomat(w2c_mat, points=pcd)

    import wrs.visualization.panda.world as wd
    from wrs.drivers.devices.realsense.realsense_d400s import RealSenseD400
    from wrs.robot_sim.robots.xarmlite6_wg import x6wg2

    base = wd.World(cam_pos=rm.vec(2, 0, 1.5), lookat_pos=rm.vec(0, 0, 0))
    rs_pipe = RealSenseD400()
    # the first frame contains no data information
    rs_pipe.get_pcd_texture_depth()
    rs_pipe.get_pcd_texture_depth()
    # rbtx = XArmLite6X(ip='192.168.1.190', has_gripper=False)
    rbt = x6wg2.XArmLite6WG2()

    xarm_mc = XArmLite6ManualCalib(rbt_s=rbt, rbt_x=None, sensor_hdl=rs_pipe)
    base.run()

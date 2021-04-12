import numpy as np
import basis.robot_math as rm
import motion.optimization_based.incremental_nik as inik

class DepthCaliberator(object):

    def __init__(self, robot_x, sensor_client):
        self.robot_x = robot_x
        self.robot_inik_solver = inik.IncrementalNIK(robot_instance)
        self.sensor_client = sensor_client

    def _find_tcp_in_sensor(self, component_name, action_pos, action_rotmat, aruco_info, criteria_radius = None):
        """
        find the robot_s tcp's pos and rotmat in the sensor coordinate system
        :param component_name:
        :param action_pos, action_rotmat:
        :param aruco_info:
        :param criteria_radius: rough radius used to determine if the newly estimated center is correct or not
        :return:
        author: weiwei
        date: 20210408
        """

        def _fit_sphere(p, coords):
            x0, y0, z0, R = p
            x, y, z = coords.T
            return np.sqrt((x-x0)**2 + (y-y0)**2 + (z-z0)**2)
        _err_fit_sphere = lambda p, x: _fit_sphere(p, x) - p[3]

        coords = []
        rot_range_x = [np.array([1,0,0]), [-30,-15,0,15,30]]
        rot_range_y = [np.array([0,1,0]), [-30,-15,15,30]]
        rot_range_z = [np.array([0,0,1]), [-90,-60,-30,30,60]]
        range_axes = [rot_range_x, rot_range_y, rot_range_z]
        lastarmjnts = self.robot_x.lft_arm_hnd.get_jnt_values()
        for axisid in range(3):
            axis = range_axes[axisid][0]
            for angle in range_axes[axisid][1]:
                goal_pos = action_pos
                goal_rot = np.dot(rm.rotmat_from_axangle(axis, angle), action_rotmat)
                armjnts = yhx.movetoposrotmsc(eepos=goal_pos, eerot=goal_rot, msc=lastarmjnts, armname=armname)
                if armjnts is not None and not yhx.pcdchecker.isSelfCollided(yhx.rbt):
                    lastarmjnts = armjnts
                    yhx.movetox(armjnts, armname=armname)
                    pxc.triggerframe()
                    img = pxc.gettextureimg()
                    pcd = pxc.getpcd()
                    phxpos = getcenter(img, pcd, aruco_dict, parameters)
                    if phxpos is not None:
                        coords.append(phxpos)
        print(coords)
        if len(coords) < 3:
            return [None, None]
        # for coord in coords:
        #     yhx.p3dh.gensphere(coord, radius=5).reparentTo(base.render)
        coords = np.asarray(coords)
        # try:
        initialguess = np.ones(4)
        initialguess[:3] = np.mean(coords, axis=0)
        finalestimate, flag = leastsq(errfunc, initialguess, args=(coords,))
        if len(finalestimate) == 0:
            return [None, None]
        print(finalestimate)
        print(np.linalg.norm(coords - finalestimate[:3], axis=1))
        # yhx.p3dh.gensphere(finalestimate[:3], rgba=np.array([0,1,0,1]), radius=5).reparentTo(base.render)
        # yhx.base.run()
        # except:
        #     return [None, None]
        if criteriaradius is not None:
            if abs(finalestimate[3]-criteriaradius) > 5:
                return [None, None]
        return np.array(finalestimate[:3]), finalestimate[3]


    def find_board_center_in_hand(self, component_name, parameters,):
        pass
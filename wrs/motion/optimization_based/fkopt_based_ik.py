import math
import time
import numpy as np
from scipy.optimize import minimize
import wrs.basis.robot_math as rm


class FKOptBasedIK(object):

    def __init__(self, robot, component_name, obstacle_list=[], toggle_debug=False):
        self.rbt = robot
        self.jlc_name = component_name
        self.result = None
        self.seed_jnt_values = None
        self.tgt_pos = None
        self.tgt_rotmat = None
        self.cons = []
        self.bnds = self._get_bnds(jlc_name=self.jlc_name)
        self.jnts = []
        self.jnt_diff = []
        self.xangle_err = []
        self.zangle_err = []
        self.x_err = []
        self.y_err = []
        self.z_err = []
        self._xangle_limit = math.pi/360 # less than .5 degree
        self._zangle_limit = math.pi/6
        self._x_limit = 1e-6
        self._y_limit = 1e-6
        self._z_limit = 1e-6
        self.obstacle_list = obstacle_list
        self.toggle_debug = toggle_debug

    def _get_bnds(self, jlc_name):
        return self.rbt.get_jnt_rngs(jlc_name)

    def _constraint_zangle(self, jnt_values):
        self.rbt.fk(joint_values=jnt_values, component_name=self.jlc_name)
        gl_tcp_pos, gl_tcp_rotmat = self.rbt.get_gl_tcp(manipulator_name=self.jlc_name)
        delta_angle = rm.angle_between_vectors(gl_tcp_rotmat[:,2], self.tgt_rotmat[:,2])
        self.zangle_err.append(delta_angle)
        return self._zangle_limit-delta_angle

    def _constraint_xangle(self, jnt_values):
        self.rbt.fk(joint_values=jnt_values, component_name=self.jlc_name)
        gl_tcp_pos, gl_tcp_rotmat = self.rbt.get_gl_tcp(manipulator_name=self.jlc_name)
        delta_angle = rm.angle_between_vectors(gl_tcp_rotmat[:,0], self.tgt_rotmat[:,0])
        self.xangle_err.append(delta_angle)
        return self._xangle_limit-delta_angle

    def _constraint_x(self, jnt_values):
        self.rbt.fk(joint_values=jnt_values, component_name=self.jlc_name)
        gl_tcp_pos, gl_tcp_rot = self.rbt.get_gl_tcp(manipulator_name=self.jlc_name)
        x_err = abs(self.tgt_pos[0] - gl_tcp_pos[0])
        self.x_err.append(x_err)
        return self._x_limit - x_err

    def _constraint_y(self, jnt_values):
        self.rbt.fk(joint_values=jnt_values, component_name=self.jlc_name)
        gl_tcp_pos, gl_tcp_rot = self.rbt.get_gl_tcp(manipulator_name=self.jlc_name)
        y_err = abs(self.tgt_pos[1] - gl_tcp_pos[1])
        self.y_err.append(y_err)
        return self._y_limit - y_err

    def _constraint_z(self, jnt_values):
        self.rbt.fk(joint_values=jnt_values, component_name=self.jlc_name)
        gl_tcp_pos, gl_tcp_rot = self.rbt.get_gl_tcp(manipulator_name=self.jlc_name)
        z_err = abs(self.tgt_pos[2] - gl_tcp_pos[2])
        self.z_err.append(z_err)
        return self._z_limit - z_err

    def _constraint_collision(self, jnt_values):
        self.rbt.fk(joint_values=jnt_values, component_name=self.jlc_name)
        if self.rbt.is_collided(obstacle_list=self.obstacle_list):
            return -1
        else:
            return 1

    def add_constraint(self, fun, type="ineq"):
        self.cons.append({'end_type': type, 'fun': fun})

    def optimization_goal(self, jnt_values):
        if self.toggle_debug:
            self.jnts.append(jnt_values)
            self.jnt_diff.append(np.linalg.norm(self.seed_jnt_values - jnt_values))
            # if random.choice(range(20)) == 0:
            #     self.rbth.show_armjnts(armjnts=self.joints[-1], rgba=(.7, .7, .7, .2))
        return np.linalg.norm(jnt_values - self.seed_jnt_values)

    def solve(self, tgt_pos, tgt_rotmat, seed_jnt_values, method='SLSQP'):
        """
        :param tgt_pos:
        :param tgt_rotmat:
        :param seed_jnt_values:
        :param method:
        :return:
        """
        self.seed_jnt_values = seed_jnt_values
        self.tgt_pos = tgt_pos
        self.tgt_rotmat = tgt_rotmat
        self.add_constraint(self._constraint_xangle, type="ineq")
        self.add_constraint(self._constraint_zangle, type="ineq")
        self.add_constraint(self._constraint_x, type="ineq")
        self.add_constraint(self._constraint_y, type="ineq")
        self.add_constraint(self._constraint_z, type="ineq")
        self.add_constraint(self._constraint_collision, type="ineq")
        time_start = time.time()
        sol = minimize(self.optimization_goal,
                       seed_jnt_values,
                       method=method,
                       bounds=self.bnds,
                       constraints=self.cons)
        print("time cost", time.time() - time_start)
        if self.toggle_debug:
            print(sol)
            self._debug_plot()
        if sol.success:
            return sol.x, sol.fun
        else:
            return None, None

    def gen_linear_motion(self,
                          start_info,
                          goal_info,
                          granularity=0.03,
                          seed_jnt_values=None):
        """
        :param start_info:
        :param goal_info:
        :param granularity:
        :return:
        author: weiwei
        date: 20210125
        """
        jnt_values_bk = self.rbt.get_jnt_values(self.jlc_name)
        pos_list, rotmat_list = rm.interplate_pos_rotmat(start_info[0],
                                                         start_info[1],
                                                         goal_info[0],
                                                         goal_info[1],
                                                         granularity=granularity)
        jnt_values_list = []
        if seed_jnt_values is None:
            seed_jnt_values = jnt_values_bk
        for (pos, rotmat) in zip(pos_list, rotmat_list):
            jnt_values = self.solve(pos, rotmat, seed_jnt_values)[0]
            if jnt_values is None:
                print("Not solvable!")
                self.rbt.fk(self.jlc_name, jnt_values_bk)
                return []
            jnt_values_list.append(jnt_values)
            seed_jnt_values = jnt_values
        self.rbt.fk(self.jlc_name, jnt_values_bk)
        return jnt_values_list

    def _debug_plot(self):
        if "plt" not in dir():
            import wrs.visualization.matplot.helper as plth
        plth.plt.figure(1, figsize=(6.4 * 3, 4.8 * 2))
        plth.plt.subplot(231)
        plth.plt.plot(*plth.list_to_plt_xy(self.x_err))
        plth.plt.subplot(232)
        plth.plt.plot(*plth.list_to_plt_xy(self.y_err))
        plth.plt.subplot(233)
        plth.plt.plot(*plth.list_to_plt_xy(self.z_err))
        plth.plt.subplot(234)
        plth.plt.plot(*plth.list_to_plt_xy(self.xangle_err))
        plth.plt.subplot(235)
        plth.plt.plot(*plth.list_to_plt_xy(self.zangle_err))
        plth.plt.subplot(236)
        plth.plt.plot(*plth.twodlist_to_plt_xys(self.jnts))
        # plth.plot_list(self.rot_err, title="rotation error")
        # plth.plot_list(self.jnt_diff, title="joints displacement")
        plth.plt.show()

if __name__ == '__main__':
    import wrs.visualization.panda.world as wd
    import wrs.modeling.geometric_model as mgm
    import wrs.robot_sim.robots.yumi.yumi as ym

    base = wd.World(cam_pos=[1.5, 0, 3], lookat_pos=[0, 0, .5])
    component_name= 'rgt_arm'
    tgt_pos = np.array([.5, -.3, .3])
    tgt_rotmat = rm.rotmat_from_axangle([0,1,0], math.pi/2)
    mgm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
    yumi_instance = ym.Yumi(enable_cc=True)
    oik = FKOptBasedIK(yumi_instance, component_name=component_name, toggle_debug=False)
    # jnt_values, _ = oik.solve(tgt_pos, tgt_rotmat, np.zeros(7), method='SLSQP')
    # print(jnt_values)
    # robot_s.fk(hnd_name=hnd_name, jnt_values=jnt_values)
    # yumi_meshmodel = robot_s.gen_meshmodel()
    # yumi_meshmodel.attach_to(base)
    start_pos = np.array([.5, -.3, .3])
    start_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi / 2)
    start_info = [start_pos, start_rotmat]
    goal_pos = np.array([.6, .1, .5])
    goal_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi / 2)
    goal_info = [goal_pos, goal_rotmat]
    mgm.gen_frame(pos=start_pos, rotmat=start_rotmat).attach_to(base)
    mgm.gen_frame(pos=goal_pos, rotmat=goal_rotmat).attach_to(base)
    tic = time.time()
    jnt_values_list = oik.gen_linear_motion(start_info, goal_info)
    toc = time.time()
    print(toc - tic)
    for jnt_values in jnt_values_list:
        yumi_instance.fk(component_name, jnt_values)
        yumi_meshmodel = yumi_instance.gen_meshmodel()
        yumi_meshmodel.attach_to(base)
    base.run()
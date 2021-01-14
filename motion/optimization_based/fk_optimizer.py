import copy
import math
import time
import random
import numpy as np
import warnings as wns
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import basis.robotmath as rm


class FKOptimizer(object):

    def __init__(self, robot, jlc_name, toggle_debug=False):
        self.rbt = robot
        self.jlc_name = jlc_name
        self.result = None
        self.seed_jnt_values = None
        self.tgt_pos = None
        self.tgt_rotmat = None
        self.cons = []
        self.bnds = self._get_bnds(jlc_name=self.jlc_name)
        self.jnts = []
        self.jnt_diff = []
        self.roll_err = []
        self.pitch_err = []
        self.yaw_err = []
        self.x_err = []
        self.y_err = []
        self.z_err = []
        self._roll_limit = 1e-6
        self._pitch_limit = 1e-6
        self._yaw_limit = 1e-6
        self._x_limit = 1e-6
        self._y_limit = 1e-6
        self._z_limit = 1e-6
        self.toggle_debug = toggle_debug

    def _get_bnds(self, jlc_name):
        return self.rbt.get_jnt_ranges(jlc_name)

    def _constraint_roll(self, jnt_values):
        self.rbt.fk(jnt_values=jnt_values, component_name=self.jlc_name)
        gl_tcp_pos, gl_tcp_rot = self.rbt.get_gl_tcp(component_name=self.jlc_name)
        angle = rm.angle_between_vectors(gl_tcp_rot[:3,0], self.tgt_rotmat[:3,0])
        self.roll_err.append(angle)
        return self._roll_limit-angle

    def _constraint_pitch(self, jnt_values):
        self.rbt.fk(jnt_values=jnt_values, component_name=self.jlc_name)
        gl_tcp_pos, gl_tcp_rot = self.rbt.get_gl_tcp(component_name=self.jlc_name)
        angle = rm.angle_between_vectors(gl_tcp_rot[:3,1], self.tgt_rotmat[:3,1])
        self.pitch_err.append(angle)
        return self._roll_limit-angle

    def _constraint_yaw(self, jnt_values):
        self.rbt.fk(jnt_values=jnt_values, component_name=self.jlc_name)
        gl_tcp_pos, gl_tcp_rot = self.rbt.get_gl_tcp(component_name=self.jlc_name)
        angle = rm.angle_between_vectors(gl_tcp_rot[:3,2], self.tgt_rotmat[:3,2])
        self.yaw_err.append(angle)
        return self._roll_limit-angle

    def _constraint_x(self, jnt_values):
        self.rbt.fk(jnt_values=jnt_values, component_name=self.jlc_name)
        gl_tcp_pos, gl_tcp_rot = self.rbt.get_gl_tcp(component_name=self.jlc_name)
        x_err = abs(self.tgt_pos[0] - gl_tcp_pos[0])
        self.x_err.append(x_err)
        return self._x_limit - x_err

    def _constraint_y(self, jnt_values):
        self.rbt.fk(jnt_values=jnt_values, component_name=self.jlc_name)
        gl_tcp_pos, gl_tcp_rot = self.rbt.get_gl_tcp(component_name=self.jlc_name)
        y_err = abs(self.tgt_pos[0] - gl_tcp_pos[0])
        self.y_err.append(y_err)
        return self._y_limit - y_err

    def _constraint_z(self, jnt_values):
        self.rbt.fk(jnt_values=jnt_values, component_name=self.jlc_name)
        gl_tcp_pos, gl_tcp_rot = self.rbt.get_gl_tcp(component_name=self.jlc_name)
        z_err = abs(self.tgt_pos[0] - gl_tcp_pos[0])
        self.z_err.append(z_err)
        return self._z_limit - z_err

    def _constraint_collision(self, jnt_values):
        self.rbt.fk(jnt_values=jnt_values, component_name=self.jlc_name)
        if self.rbt.is_collided():
            return -1
        else:
            return 1

    def _add_constraint(self, constraint, condition="ineq"):
        self.cons.append({'type': condition, 'fun': constraint})

    def optimization_goal(self, jnt_values):
        if self.toggle_debug:
            self.jnts.append(jnt_values)
            self.jnt_diff.append(np.linalg.norm(self.seed_jnt_values - jnt_values))
            # if random.choice(range(20)) == 0:
            #     self.rbth.show_armjnts(armjnts=self.jnts[-1], rgba=(.7, .7, .7, .2))
        return np.linalg.norm(jnt_values - self.seed_jnt_values)

    def solve(self, seed_jnt_values, tgt_pos, tgt_rotmat=None, method='SLSQP'):
        """
        :param seed_jnt_values:
        :param tgt_pos:
        :param tgt_rotmat:
        :param method:
        :return:
        """
        self.seed_jnt_values = seed_jnt_values
        self.tgt_pos = tgt_pos
        self.tgt_rotmat = tgt_rotmat
        # self._add_constraint(self._constraint_roll, condition="ineq")
        # self._add_constraint(self._constraint_pitch, condition="ineq")
        # self._add_constraint(self._constraint_yaw, condition="ineq")
        self._add_constraint(self._constraint_x, condition="ineq")
        self._add_constraint(self._constraint_y, condition="ineq")
        self._add_constraint(self._constraint_z, condition="ineq")
        # self._add_constraint(self.con_collision, condition="ineq")
        time_start = time.time()
        # iks = IkSolver(self.env, self.rbt, self.rbtmg, self.rbtball, self.armname)
        # q0 = iks.slove_numik3(self.tgtpos, tgtrot=None, seed_jnt_values=self.seed_jnt_values, releemat4=self.releemat4)
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

    def _debug_plot(self):
        fig = plt.figure(1, figsize=(6.4 * 3, 4.8 * 2))
        plt.subplot(231)
        self.rbth.plot_vlist(self.pos_err, title="translation error")
        plt.subplot(232)
        self.rbth.plot_vlist(self.rot_err, title="rotation error")
        plt.subplot(233)
        self.rbth.plot_vlist(self.jnt_diff, title="jnts displacement")
        plt.subplot(234)
        self.rbth.plot_armjnts(self.jnts, show=False)
        plt.show()

if __name__ == '__main__':
    import visualization.panda.world as wd
    import robotsim.robots.yumi.yumi as ym
    import modeling.geometricmodel as gm

    base = wd.World(campos=[1.5, 0, 3], lookatpos=[0, 0, .5])
    jlc_name='rgt_arm'
    tgt_pos = np.array([.5, -.3, .3])
    tgt_rotmat = rm.rotmat_from_axangle([0,1,0], math.pi/2)
    gm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
    yumi_instance = ym.Yumi(enable_cc=True)
    oik = FKOptimizer(yumi_instance, jlc_name=jlc_name, toggle_debug=False)
    jnt_values, _ = oik.solve(np.zeros(7), tgt_pos, tgt_rotmat=tgt_rotmat, method='SLSQP')
    print(jnt_values)
    yumi_instance.fk(jnt_values, component_name=jlc_name)
    yumi_meshmodel = yumi_instance.gen_meshmodel()
    yumi_meshmodel.attach_to(base)
    base.run()
import time
import math
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from operator import itemgetter
from scipy.optimize import minimize
from scipy.optimize import Bounds


# NOTE: write your own extend_state_callback and goal_test_callback to implement your own kinodyanmics
class Kinodynamics(object):
    def __init__(self, time_interval=.1):
        self.linear_speed_rng = [-1.0, 1.0]
        self.angular_speed_rng = [-.5, .5]
        self.linear_acc = 1.0
        self.angular_acc = 3.5
        self.time_interval = time_interval
        self.weights = np.array([1, .1, 0, 0])
        self.epsilon = 1e-3

    def annihilator(self, theta_value):
        return np.array([[math.cos(theta_value), math.sin(theta_value), 0],
                         [0, 0, 1]])

    def _goal_function(self, x):
        new_state = np.zeros_like(self._state1)
        # new_state_angle = self._state1[2] + self._state1[5] * self.time_intervals
        new_state_angle = self._state1[2] + (self._state1[5] + x[1]) / 2 * self.time_interval
        new_state[3:] = x.dot(self.annihilator(new_state_angle))
        new_state[:3] = self._state1[:3] + (self._state1[3:] + new_state[3:]) / 2 * self.time_interval
        return_value = self.metric(new_state, self._state2)
        return return_value

    def metric(self, state1, state2):
        diff_state = state1 - state2
        measurement = np.array([np.linalg.norm(diff_state[:2]),
                                min(abs(diff_state[2]), abs(diff_state[2] + 2 * math.pi),
                                    abs(diff_state[2] - 2 * math.pi)),
                                np.linalg.norm(diff_state[3:5]),
                                np.abs(diff_state[5])])
        return self.weights.dot(measurement)

    def set_goal_state(self, goal_state):
        self._goal_state = goal_state

    def extend_state_callback(self, state1, state2):
        """
        extend state call back for two-wheel car rbt_s
        :param state1: x, y, theta, x_dot, y_dot, theta_dot
        :param state2:
        :return:
        """
        self._state1 = state1
        self._state2 = state2
        s1_ls = np.linalg.norm(state1[3:5])  # linear speed at state 1
        s1_as = state1[5]  # angular speed at state 1
        if np.sign(math.cos(state1[2])) != np.sign(state1[3]):
            s1_ls = -s1_ls
        x_bnds = Bounds(lb=[self.linear_speed_rng[0], self.angular_speed_rng[0]],
                        ub=[self.linear_speed_rng[1], self.angular_speed_rng[1]])
        # optmize the ns_bnds for t+1
        # acc constraints
        ineq_cons = {'end_type': 'ineq',
                     'fun': lambda x: np.array([self.linear_acc ** 2 - ((x[0] - s1_ls) / self.time_interval) ** 2,
                                                self.angular_acc ** 2 - ((x[1] - s1_as) / self.time_interval) ** 2])}
        x0 = np.array([s1_ls, s1_as])
        res = minimize(self._goal_function, x0,
                       method='SLSQP', constraints=[ineq_cons],
                       options={'ftol': self.epsilon, 'disp': True},
                       bounds=x_bnds)
        return_state = np.zeros_like(state1)
        # return_state_angle = state1[2] + state1[5] * self.time_intervals
        return_state_angle = state1[2] + (state1[5] + res.x[1]) / 2 * self.time_interval
        return_state[3:] = res.x.dot(self.annihilator(return_state_angle))
        return_state[:3] = state1[:3] + (state1[3:] + return_state[3:]) / 2 * self.time_interval
        current_metric = self.metric(state1, state2)
        new_metric = self.metric(return_state, state2)
        print("control ", res.x)
        print("this ", state1)
        print("next ", return_state)
        print("rand ", state2)
        print("dist this to rand", self.metric(state1, state2))
        print("dist next to rand", self.metric(return_state, state2))
        if current_metric < new_metric + self.epsilon:
            return None
        else:
            return return_state

    def goal_test_callback(self, state, goal_state):
        goal_dist = self.metric(state, goal_state)
        if goal_dist < 1e-2:
            return True
        else:
            return False


class RRTKinodynamic(object):

    def __init__(self, robot_s, kds):
        """
        :param robot_s:
        :param extend_conf_callback: call back function for extend_conf
        """
        self.robot_s = robot_s.copy()
        self.roadmap = nx.Graph()
        self.start_conf = None
        self.goal_conf = None
        self.roadmap = nx.DiGraph()
        self.kds = kds

    def _is_collided(self,
                     component_name,
                     conf,
                     obstacle_list=[],
                     otherrobot_list=[]):
        self.robot_s.fk(component_name=component_name, joint_values=conf)
        return self.robot_s.is_collided(obstacle_list=obstacle_list, other_robot_list=otherrobot_list)

    def _sample_conf(self, component_name, rand_rate, default_conf):
        rand_number = np.random.uniform(0, 100.0)
        print("random number/rate: ", rand_number, rand_rate)
        if rand_number < rand_rate:
            rand_conf = self.robot_s.rand_conf(component_name=component_name)
            rand_ls = np.random.uniform(self.kds.linear_speed_rng[0], self.kds.linear_speed_rng[1])
            rand_as = np.random.uniform(self.kds.angular_speed_rng[0], self.kds.angular_speed_rng[1])
            rand_speed = np.array([rand_ls, rand_as]).dot(self.kds.annihilator(rand_conf[2]))
            return np.hstack((rand_conf, rand_speed))
        else:
            return default_conf

    def _get_nearest_nid(self, roadmap, new_state):
        """
        convert to numpy to accelerate access
        :param roadmap:
        :param new_state:
        :return:
        author: weiwei
        date: 20210523
        """
        nodes_dict = dict(roadmap.nodes(data='conf'))
        nodes_key_list = list(nodes_dict.keys())
        nodes_value_list = list(nodes_dict.values())
        state_array = np.array(nodes_value_list)
        # diff_conf_array = np.linalg.norm(conf_array[:,:self.kds.conf_dof] - new_state[:self.kds.conf_dof], axis=1)
        diff_state = state_array - new_state
        tmp0 = np.abs(diff_state[:, 2])
        tmp1 = np.abs(diff_state[:, 2] + 2 * math.pi)
        tmp2 = np.abs(diff_state[:, 2] - 2 * math.pi)
        diff_state_array = self.kds.weights[0] * np.linalg.norm(diff_state[:, :2], axis=1) + \
                           self.kds.weights[1] * np.min(np.vstack((tmp0, tmp1, tmp2))).T + \
                           self.kds.weights[2] * np.linalg.norm(diff_state[:, 3:5], axis=1) + \
                           self.kds.weights[3] * np.abs(diff_state[:, 5])
        min_dist_nid = np.argmin(diff_state_array)
        return nodes_key_list[min_dist_nid]

    # def _extend_roadmap(self,
    #                     component_name,
    #                     roadmap,
    #                     conf,
    #                     goal_state,
    #                     obstacle_list=[],
    #                     other_robot_list=[],
    #                     animation=False):
    #     """
    #     find the nearest point between the given roadmap and the conf and then extend towards the conf
    #     :return:
    #     author: weiwei
    #     date: 20201228
    #     """
    #     nearest_nid = self._get_nearest_nid(roadmap, conf)
    #     new_state = self.kds.extend_state_callback(roadmap.nodes[nearest_nid]['conf'], conf)
    #     print("near state ", roadmap.nodes[nearest_nid]['conf'])
    #     print("new state ", new_state)
    #     if new_state is not None:
    #         if self._is_collided(component_name, new_state, obstacle_list, other_robot_list):
    #             return nearest_nid
    #         else:
    #             new_nid = random.randint(0, 1e8)
    #             roadmap.add_node(new_nid, conf=new_state)
    #             roadmap.add_edge(nearest_nid, new_nid)
    #             # all_sampled_confs.append([new_node.point, False])
    #             if animation:
    #                 self.draw_sspace([roadmap], self.start_state, self.goal_state,
    #                                  obstacle_list, [roadmap.nodes[nearest_nid]['conf'], conf],
    #                                  new_state, None)
    #             # check goal
    #             if self.kds.goal_test_callback(roadmap.nodes[new_nid]['conf'], goal_state):
    #                 roadmap.add_node('connection', conf=goal_state)  # TODO current name -> connection
    #                 roadmap.add_edge(new_nid, 'connection')
    #                 return 'connection'
    #             return new_nid
    #     else:
    #         return nearest_nid

    def _extend_roadmap(self,
                        component_name,
                        roadmap,
                        conf,
                        goal_conf,
                        obstacle_list=[],
                        otherrobot_list=[],
                        animation=False):
        """
        find the nearest point between the given roadmap and the conf and then extend towards the conf
        :return:
        author: weiwei
        date: 20201228
        """
        nearest_nid = self._get_nearest_nid(roadmap, conf)
        while True:
            new_state = self.kds.extend_state_callback(roadmap.nodes[nearest_nid]['conf'], conf)
            print("near state ", roadmap.nodes[nearest_nid]['conf'])
            print("new state ", new_state)
            if new_state is not None:
                if self._is_collided(component_name, new_state, obstacle_list, otherrobot_list):
                    return nearest_nid
                else:
                    new_nid = random.randint(0, 1e12)
                    roadmap.add_node(new_nid, conf=new_state)
                    roadmap.add_edge(nearest_nid, new_nid)
                    # all_sampled_confs.append([new_node.point, False])
                    if animation:
                        self.draw_sspace([roadmap], self.start_conf, self.goal_conf,
                                         obstacle_list, [roadmap.nodes[nearest_nid]['conf'], conf],
                                         new_state, None)
                    # check goal
                    if self.kds.goal_test_callback(roadmap.nodes[new_nid]['conf'], goal_conf):
                        roadmap.add_node('connection', conf=goal_conf)  # TODO current name -> connection
                        roadmap.add_edge(new_nid, 'connection')
                        return 'connection'
                    nearest_nid = new_nid
            else:
                return nearest_nid

    def _path_from_roadmap(self):
        nid_path = nx.shortest_path(self.roadmap, 'start', 'goal')
        return list(itemgetter(*nid_path)(self.roadmap.nodes(data='conf')))

    def plan(self,
             component_name,
             start_state,
             goal_conf,
             obstacle_list=[],
             otherrobot_list=[],
             rand_rate=70,
             max_iter=10000,
             max_time=15.0,
             smoothing_iterations=17,
             animation=False):
        """
        :return: [path, all_sampled_confs]
        author: weiwei
        date: 20201226
        """
        self.roadmap.clear()
        self.start_conf = start_state
        self.goal_conf = goal_conf
        # check seed_jnt_values and end_conf
        if self._is_collided(component_name, start_state, obstacle_list, otherrobot_list):
            print("The start robot_s configuration is in collision!")
            return None
        if self._is_collided(component_name, goal_conf, obstacle_list, otherrobot_list):
            print("The goal robot_s configuration is in collision!")
            return None
        if self.kds.goal_test_callback(state=start_state, goal_state=goal_conf):
            return [[start_state, goal_conf], None]
        self.roadmap.add_node('start', conf=start_state, cost=0)
        self.kds.set_goal_state(goal_conf)
        tic = time.time()
        for _ in range(max_iter):
            toc = time.time()
            if max_time > 0.0:
                if toc - tic > max_time:
                    print("Too much motion time! Failed to find a path.")
                    return None
            # Random Sampling
            rand_conf = self._sample_conf(component_name=component_name, rand_rate=rand_rate, default_conf=goal_conf)
            last_nid = self._extend_roadmap(component_name=component_name,
                                            roadmap=self.roadmap,
                                            conf=rand_conf,
                                            goal_conf=goal_conf,
                                            obstacle_list=obstacle_list,
                                            otherrobot_list=otherrobot_list,
                                            animation=animation)
            if last_nid == 'connection':
                mapping = {'connection': 'goal'}
                self.roadmap = nx.relabel_nodes(self.roadmap, mapping)
                path = self._path_from_roadmap()
                return path
        else:
            print("Reach to maximum iteration! Failed to find a path.")
            return None

    @staticmethod
    def draw_sspace(roadmap_list,
                    start_state,
                    goal_state,
                    obstacle_list,
                    near_rand_state_pair=None,
                    new_state=None,
                    new_state_list=None,
                    shortcut=None,
                    smoothed_path=None,
                    delay_time=.02):

        def draw_robot(state, facecolor='grey', edgecolor='grey'):
            x = state[0]
            y = state[1]
            theta = state[2]
            ax.add_patch(plt.Circle((x,y), .5, edgecolor=edgecolor, facecolor=facecolor))
            ax.add_patch(plt.Rectangle((x,y), .7, .1, math.degrees(theta), color='y'))
            ax.add_patch(plt.Rectangle((x,y), -.1, .1, math.degrees(theta),
                                       edgecolor=edgecolor, facecolor=facecolor))
            ax.add_patch(plt.Rectangle((x,y), .7, -.1, math.degrees(theta), color='y'))
            ax.add_patch(plt.Rectangle((x,y), -.1, -.1, math.degrees(theta),
                                       edgecolor=edgecolor, facecolor=facecolor))

        plt.clf()
        ax = plt.gca()
        ax.set_aspect('equal', 'box')
        plt.grid(True)
        plt.xlim(-4.0, 17.0)
        plt.ylim(-4.0, 17.0)
        draw_robot(start_state, facecolor='r', edgecolor='r')
        draw_robot(goal_state, facecolor='g', edgecolor='g')
        for (point, size) in obstacle_list:
            ax.add_patch(plt.Circle((point[0], point[1]), size / 2.0, color='k'))
        colors = 'bgrcmykw'
        for i, roadmap in enumerate(roadmap_list):
            for (u, v) in roadmap.edges:
                plt.plot(roadmap.nodes[u]['conf'][0], roadmap.nodes[u]['conf'][1], 'o' + colors[i])
                plt.plot(roadmap.nodes[v]['conf'][0], roadmap.nodes[v]['conf'][1], 'o' + colors[i])
                plt.plot([roadmap.nodes[u]['conf'][0], roadmap.nodes[v]['conf'][0]],
                         [roadmap.nodes[u]['conf'][1], roadmap.nodes[v]['conf'][1]], '-' + colors[i])
        if near_rand_state_pair is not None:
            plt.plot([near_rand_state_pair[0][0], near_rand_state_pair[1][0]],
                     [near_rand_state_pair[0][1], near_rand_state_pair[1][1]], "--k")
            draw_robot(near_rand_state_pair[0], facecolor='grey', edgecolor='g')
            draw_robot(near_rand_state_pair[1], facecolor='grey', edgecolor='c')
        if new_state is not None:
            draw_robot(new_state, facecolor='grey', edgecolor='c')
        if new_state_list is not None:
            for new_state in new_state_list:
                plt.plot(new_state[0], new_state[1], 'or')
                plt.plot([new_state[0], near_rand_state_pair[0][0]],
                         [new_state[1], near_rand_state_pair[0][1]], '--r')
        if smoothed_path is not None:
            plt.plot([conf[0] for conf in smoothed_path], [conf[1] for conf in smoothed_path], linewidth=7,
                     linestyle='-', color='c')
        if shortcut is not None:
            plt.plot([conf[0] for conf in shortcut], [conf[1] for conf in shortcut], linewidth=4, linestyle='--',
                     color='r')
        # plt.plot(planner.seed_jnt_values[0], planner.seed_jnt_values[1], "xr")
        # plt.plot(planner.end_conf[0], planner.end_conf[1], "xm")
        if not hasattr(RRTKinodynamic, 'img_counter'):
            RRTKinodynamic.img_counter = 0
        else:
            RRTKinodynamic.img_counter += 1
        # plt.savefig(str( RRT.img_counter)+'.jpg')
        if delay_time > 0:
            plt.pause(delay_time)
        # plt.waitforbuttonpress()


if __name__ == '__main__':
    from wrs import robot_sim as xyb

    # ====Search Path with RRT====
    obstacle_list = [
        ((5, 5), 3),
        ((3, 6), 3),
        ((3, 8), 3),
        ((3, 10), 3),
        ((7, 5), 3),
        ((9, 5), 3),
        ((10, 5), 3)
    ]  # [x,y,size]
    # Set Initial parameters
    robot_s = xyb.XYWBot()
    kds = Kinodynamics(time_interval=.5)
    rrtkino_s = RRTKinodynamic(robot_s, kds)
    path = rrtkino_s.plan(start_state=np.array([.0, .0, .0, .0, .0, .0]),
                          goal_conf=np.array([6.0, 9.0, .0, .0, .0, .0]),
                          obstacle_list=obstacle_list,
                          rand_rate=10, max_time=1000,
                          component_name='all', smoothing_iterations=0,
                          animation=True)
    # plt.show()
    # nx.draw(rrt.roadmap, with_labels=True, font_weight='bold')
    # plt.show()
    # import time
    # total_t = 0
    # for i in range(1):
    #     tic = time.time()
    #     path, sampledpoints = rrt.motion(obstaclelist=obstaclelist, animation=True)
    #     toc = time.time()
    #     total_t = total_t + toc - tic
    # print(total_t)
    # Draw final path
    print(path)
    rrtkino_s.draw_sspace([rrtkino_s.roadmap], rrtkino_s.start_conf, rrtkino_s.goal_conf, obstacle_list, delay_time=0)
    ax = plt.gca()
    for conf in path:
        ax.add_patch(plt.Rectangle((conf[0], conf[1]),
                                   .1, .3, math.degrees(conf[2]), color='y'))
        ax.add_patch(plt.Rectangle((conf[0], conf[1]),
                                   -.7, .3, math.degrees(conf[2]), edgecolor='r', facecolor='grey'))
        ax.add_patch(plt.Rectangle((conf[0], conf[1]),
                                   .1, -.3, math.degrees(conf[2]), color='y'))
        ax.add_patch(plt.Rectangle((conf[0], conf[1]),
                                   -.7, -.3, math.degrees(conf[2]), edgecolor='r', facecolor='grey'))
    # pathsm = smoother.pathsmoothing(path, rrt, 30)
    # plt.plot([point[0] for point in pathsm], [point[1] for point in pathsm], '-r')
    # plt.pause(0.001)  # Need for Mac
    plt.show()

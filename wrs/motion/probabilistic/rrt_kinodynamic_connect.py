import time
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from operator import itemgetter


# NOTE: write your own extend_state_callback and goal_test_callback to implement your own kinodyanmics
class Kinodynamics(object):
    def __init__(self, time_interval=.1):
        self.line_speed_rng = [-1.0, 1.0]
        self.angular_speed_rng = [-.5, .5]
        self.linear_acc = .7
        self.angular_acc = .5
        self.time_interval = time_interval
        self.conf_dof = 3

    def extend_state_callback(self, state1, state2):
        """
        extend state call back for two-wheel car rbt_s
        :param state1: x, y, theta, x_dot, y_dot, theta_dot
        :param state2:
        :return:
        """
        random_step_array = [[self.linear_acc * self.time_interval, 0], \
                             [-self.linear_acc * self.time_interval, 0], \
                             [0, -self.angular_acc * self.time_interval], \
                             [0, self.angular_acc * self.time_interval]]
        current_speed = np.array([np.linalg.norm(state1[3:5]), state1[5]])
        min_value = 1e12
        return_result = None
        for random_step in random_step_array:
            # random increase speed, and clip the too large ones
            next_speed = current_speed + random_step
            next_speed[0] = np.clip(next_speed[0], self.line_speed_rng[0], self.line_speed_rng[1])
            next_speed[1] = np.clip(next_speed[1], self.angular_speed_rng[0], self.angular_speed_rng[1])
            # dynamics
            avg_speed = (current_speed + next_speed) / 2
            next_angle = state1[2] + avg_speed[1] * self.time_interval
            avg_angle = (state1[2]+next_angle)/2.0
            avg_annihilating_array = np.array([[np.cos(avg_angle), np.sin(avg_angle), 0], [0, 0, 1]])
            new_state_conf = state1[:3] + (avg_speed * self.time_interval).dot(avg_annihilating_array)
            next_annihilating_array = np.array([[np.cos(next_angle), np.sin(next_angle), 0], [0, 0, 1]])
            new_state_speed = next_speed.dot(next_annihilating_array)
            new_state = np.hstack((new_state_conf, new_state_speed))
            diff_state = new_state - state2
            diff_value = np.linalg.norm(diff_state)
            if diff_value < min_value:
                min_value = diff_value
                return_result = new_state
        return return_result

    def goal_test_callback(self, state, goal_state):
        if np.all(np.abs(goal_state[:3] - state[:3]) < -np.abs(state[3:]) / 2 * self.time_interval) and \
                np.linalg.norm(state[3:5]) < self.linear_acc * self.time_interval and \
                abs(state[5]) < self.angular_acc * self.time_interval:
            return True
        else:
            return False


class RRTConnectKinodynamic(object):

    def __init__(self, robot_s, kds):
        """
        :param robot_s:
        :param extend_conf_callback: call back function for extend_conf
        """
        self.robot_s = robot_s.copy()
        self.roadmap_start = nx.Graph()
        self.roadmap_goal = nx.Graph()
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
        if random.randint(0, 99) < rand_rate:
            rand_conf = self.robot_s.rand_conf(component_name=component_name)
            # return np.hstack((rand_conf, np.zeros_like(rand_conf)))
            rand_speed = np.random.rand(3)-1
            rand_speed[:2] = rand_speed[:2]*2
            return np.hstack((rand_conf, rand_speed))
        else:
            return default_conf

    def _get_nearest_nid(self, roadmap, new_conf):
        """
        convert to numpy to accelerate access
        :param roadmap:
        :param new_conf:
        :return:
        author: weiwei
        date: 20210523
        """
        nodes_dict = dict(roadmap.nodes(data='conf'))
        nodes_key_list = list(nodes_dict.keys())
        nodes_value_list = list(nodes_dict.values())
        conf_array = np.array(nodes_value_list)
        # diff_conf_array = np.linalg.norm(conf_array[:,:self.kds.conf_dof] - new_state[:self.kds.conf_dof], axis=1)
        diff_conf_array = np.linalg.norm(conf_array - new_conf, axis=1)
        min_dist_nid = np.argmin(diff_conf_array)
        return nodes_key_list[min_dist_nid]

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
        new_conf = self.kds.extend_state_callback(roadmap.nodes[nearest_nid]['conf'], conf)
        if self._is_collided(component_name, new_conf, obstacle_list, otherrobot_list):
            return nearest_nid
        else:
            new_nid = random.randint(0, 1e16)
            roadmap.add_node(new_nid, conf=new_conf)
            roadmap.add_edge(nearest_nid, new_nid)
            nearest_nid = new_nid
            # all_sampled_confs.append([new_node.point, False])
            if animation:
                self.draw_wspace([self.roadmap_start, self.roadmap_goal], self.start_conf, self.goal_conf,
                                 obstacle_list, [roadmap.nodes[nearest_nid]['conf'], conf],
                                 new_conf, '^c')
            # check goal
            if self.kds.goal_test_callback(roadmap.nodes[new_nid]['conf'], goal_conf):
                roadmap.add_node('connection', conf=goal_conf)  # TODO current name -> connection
                roadmap.add_edge(new_nid, 'connection')
                return 'connection'
        return nearest_nid

    def _path_from_roadmap(self):
        nid_path = nx.shortest_path(self.roadmap, 'start', 'goal')
        return list(itemgetter(*nid_path)(self.roadmap.nodes(data='conf')))

    def plan(self,
             component_name,
             start_conf,
             goal_conf,
             obstacle_list=[],
             otherrobot_list=[],
             max_iter=1000,
             max_time=15.0,
             animation=False):
        """
        :return: [path, all_sampled_confs]
        author: weiwei
        date: 20201226
        """
        self.roadmap.clear()
        self.start_conf = start_conf
        self.goal_conf = goal_conf
        # check seed_jnt_values and end_conf
        if self._is_collided(component_name, start_conf, obstacle_list, otherrobot_list):
            print("The start robot_s configuration is in collision!")
            return None
        if self._is_collided(component_name, goal_conf, obstacle_list, otherrobot_list):
            print("The goal robot_s configuration is in collision!")
            return None
        if self.kds.goal_test_callback(state=start_conf, goal_state=goal_conf):
            return [[start_conf, goal_conf], None]
        self.roadmap_start.add_node('start', conf=start_conf)
        self.roadmap_goal.add_node('goal', conf=goal_conf)
        tic = time.time()
        tree_a = self.roadmap_start
        tree_b = self.roadmap_goal
        tree_a_goal_conf = self.roadmap_goal.nodes['goal']['conf']
        tree_b_goal_conf = self.roadmap_start.nodes['start']['conf']
        for _ in range(max_iter):
            toc = time.time()
            if max_time > 0.0:
                if toc - tic > max_time:
                    print("Too much motion time! Failed to find a path.")
                    return None
            # Random Sampling
            rand_conf = self._sample_conf(component_name=component_name, rand_rate=100, default_conf=None)
            last_nid = self._extend_roadmap(component_name=component_name,
                                            roadmap=tree_a,
                                            conf=rand_conf,
                                            goal_conf=tree_a_goal_conf,
                                            obstacle_list=obstacle_list,
                                            otherrobot_list=otherrobot_list,
                                            animation=animation)
            if last_nid != -1: # not trapped:
                goal_nid = last_nid
                tree_b_goal_conf = tree_a.nodes[goal_nid]['conf']
                last_nid = self._extend_roadmap(component_name=component_name,
                                                roadmap=tree_b,
                                                conf=tree_a.nodes[last_nid]['conf'],
                                                goal_conf=tree_b_goal_conf,
                                                obstacle_list=obstacle_list,
                                                otherrobot_list=otherrobot_list,
                                                animation=animation)
                if last_nid == 'connection':
                    self.roadmap = nx.compose(tree_a, tree_b)
                    self.roadmap.add_edge(last_nid, goal_nid)
                    break
                elif last_nid != -1:
                    goal_nid = last_nid
                    tree_a_goal_conf = tree_b.nodes[goal_nid]['conf']
            if tree_a.number_of_nodes() > tree_b.number_of_nodes():
                tree_a, tree_b = tree_b, tree_a
                tree_a_goal_conf, tree_b_goal_conf = tree_b_goal_conf, tree_a_goal_conf
        else:
            print("Reach to maximum iteration! Failed to find a path.")
            return None
        path = self._path_from_roadmap()
        return path

    @staticmethod
    def draw_wspace(roadmap_list,
                    start_conf,
                    goal_conf,
                    obstacle_list,
                    near_rand_conf_pair=None,
                    new_conf=None,
                    new_conf_mark='^r',
                    shortcut=None,
                    smoothed_path=None,
                    delay_time=.02):
        """
        Draw Graph
        """
        plt.clf()
        ax = plt.gca()
        ax.set_aspect('equal', 'box')
        plt.grid(True)
        plt.xlim(-4.0, 17.0)
        plt.ylim(-4.0, 17.0)
        ax.add_patch(plt.Circle((start_conf[0], start_conf[1]), .5, color='r'))
        ax.add_patch(plt.Circle((goal_conf[0], goal_conf[1]), .5, color='g'))
        for (point, size) in obstacle_list:
            ax.add_patch(plt.Circle((point[0], point[1]), size / 2.0, color='k'))
        colors = 'bgrcmykw'
        for i, roadmap in enumerate(roadmap_list):
            for (u, v) in roadmap.edges:
                plt.plot(roadmap.nodes[u]['conf'][0], roadmap.nodes[u]['conf'][1], 'o' + colors[i])
                plt.plot(roadmap.nodes[v]['conf'][0], roadmap.nodes[v]['conf'][1], 'o' + colors[i])
                plt.plot([roadmap.nodes[u]['conf'][0], roadmap.nodes[v]['conf'][0]],
                         [roadmap.nodes[u]['conf'][1], roadmap.nodes[v]['conf'][1]], '-' + colors[i])
        if near_rand_conf_pair is not None:
            plt.plot([near_rand_conf_pair[0][0], near_rand_conf_pair[1][0]],
                     [near_rand_conf_pair[0][1], near_rand_conf_pair[1][1]], "--k")
            ax.add_patch(plt.Circle((near_rand_conf_pair[1][0], near_rand_conf_pair[1][1]), .3, color='grey'))
        if new_conf is not None:
            plt.plot(new_conf[0], new_conf[1], new_conf_mark)
        if smoothed_path is not None:
            plt.plot([conf[0] for conf in smoothed_path], [conf[1] for conf in smoothed_path], linewidth=7,
                     linestyle='-', color='c')
        if shortcut is not None:
            plt.plot([conf[0] for conf in shortcut], [conf[1] for conf in shortcut], linewidth=4, linestyle='--',
                     color='r')
        # plt.plot(planner.seed_jnt_values[0], planner.seed_jnt_values[1], "xr")
        # plt.plot(planner.end_conf[0], planner.end_conf[1], "xm")
        if not hasattr(RRTConnectKinodynamic, 'img_counter'):
            RRTConnectKinodynamic.img_counter = 0
        else:
            RRTConnectKinodynamic.img_counter += 1
        # plt.savefig(str( RRT.img_counter)+'.jpg')
        if delay_time > 0:
            plt.pause(delay_time)


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
    kds = Kinodynamics(time_interval=1)
    rrtkino_s = RRTConnectKinodynamic(robot_s, kds)
    path = rrtkino_s.plan(start_conf=np.array([0, 0, 0, 0, 0, 0]), goal_conf=np.array([6, 9, 0, 0, 0, 0]),
                          obstacle_list=obstacle_list, max_time=300,
                          component_name='all',
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
    rrtkino_s.draw_wspace([rrtkino_s.roadmap], rrtkino_s.start_conf, rrtkino_s.goal_conf, obstacle_list, delay_time=0)
    plt.plot([conf[0] for conf in path], [conf[1] for conf in path], linewidth=7, linestyle='-', color='c')
    # pathsm = smoother.pathsmoothing(path, rrt, 30)
    # plt.plot([point[0] for point in pathsm], [point[1] for point in pathsm], '-r')
    # plt.pause(0.001)  # Need for Mac
    plt.show()

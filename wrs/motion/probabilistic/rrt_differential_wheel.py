import time
import math
import random
import numpy as np
from wrs import basis as rm, robot_sim as xyb
import networkx as nx
import matplotlib.pyplot as plt
from operator import itemgetter


class RRTDW(object):

    def __init__(self, robot_s):
        self.robot_s = robot_s.copy()
        self.roadmap = nx.Graph()
        self.start_conf = None
        self.goal_conf = None

    def _is_collided(self,
                     component_name,
                     conf,
                     obstacle_list=[],
                     otherrobot_list=[]):
        self.robot_s.fk(component_name=component_name, joint_values=conf)
        return self.robot_s.is_collided(obstacle_list=obstacle_list, other_robot_list=otherrobot_list)

    def _sample_conf(self, component_name, rand_rate, default_conf):
        if random.randint(0, 99) < rand_rate:
            return self.robot_s.rand_conf(component_name=component_name)
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
        nodes_value_list = list(nodes_dict.values())  # attention, correspondence is not guanranteed in python
        # use the following alternative if correspondence is bad (a bit slower), 20210523, weiwei
        # # nodes_value_list = list(nodes_dict.values())
        # nodes_value_list = itemgetter(*nodes_key_list)(nodes_dict)
        # if end_type(nodes_value_list) == np.ndarray:
        #     nodes_value_list = [nodes_value_list]
        conf_array = np.array(nodes_value_list)
        diff_conf_array = np.linalg.norm(conf_array - new_conf, axis=1)
        min_dist_nid = np.argmin(diff_conf_array)
        return nodes_key_list[min_dist_nid]

    def _extend_conf(self, conf1, conf2, ext_dist):
        """
        WARNING: This extend_conf is specially designed for differential-wheel robots
        :param conf1:
        :param conf2:
        :param ext_dist:
        :return: a list of 1xn nparray
        author: weiwei
        date: 20210530
        """
        angle_ext_dist = ext_dist
        len, vec = rm.unit_vector(conf2[:2] - conf1[:2], toggle_length=True)
        if len > 0:
            translational_theta = rm.angle_between_2d_vecs(np.array([1, 0]), vec)
            conf1_theta_to_translational_theta = translational_theta - conf1[2]
        else:
            conf1_theta_to_translational_theta = (conf2[2] - conf1[2])
            translational_theta = conf2[2]
        # rotate
        nval = abs(math.ceil(conf1_theta_to_translational_theta / angle_ext_dist))
        linear_conf1 = np.array([conf1[0], conf1[1], translational_theta])
        conf1_angular_arary = np.linspace(conf1, linear_conf1, nval)
        # translate
        nval = math.ceil(len / ext_dist)
        linear_conf2 = np.array([conf2[0], conf2[1], translational_theta])
        conf12_linear_arary = np.linspace(linear_conf1, linear_conf2, nval)
        # rotate
        translational_theta_to_conf2_theta = conf2[2] - translational_theta
        nval = abs(math.ceil(translational_theta_to_conf2_theta / angle_ext_dist))
        conf2_angular_arary = np.linspace(linear_conf2, conf2, nval)
        conf_array = np.vstack((conf1_angular_arary, conf12_linear_arary, conf2_angular_arary))
        return list(conf_array)

    def _extend_roadmap(self,
                        component_name,
                        roadmap,
                        conf,
                        ext_dist,
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
        new_conf_list = self._extend_conf(roadmap.nodes[nearest_nid]['conf'], conf, ext_dist)[1:]
        for new_conf in new_conf_list:
            if self._is_collided(component_name, new_conf, obstacle_list, otherrobot_list):
                return nearest_nid
            else:
                new_nid = random.randint(0, 1e16)
                roadmap.add_node(new_nid, conf=new_conf)
                roadmap.add_edge(nearest_nid, new_nid)
                nearest_nid = new_nid
                # all_sampled_confs.append([new_node.point, False])
                if animation:
                    self.draw_wspace([roadmap], self.start_conf, self.goal_conf,
                                     obstacle_list, [roadmap.nodes[nearest_nid]['conf'], conf],
                                     new_conf)
                # check goal
                if self._goal_test(conf=roadmap.nodes[new_nid]['conf'], goal_conf=goal_conf, threshold=ext_dist):
                    roadmap.add_node('connection', conf=goal_conf)  # TODO current name -> connection
                    roadmap.add_edge(new_nid, 'connection')
                    return 'connection'
        else:
            return nearest_nid

    def _goal_test(self, conf, goal_conf, threshold):
        dist = np.linalg.norm(conf - goal_conf)
        if dist <= threshold:
            # print("Goal reached!")
            return True
        else:
            return False

    def _path_from_roadmap(self):
        nid_path = nx.shortest_path(self.roadmap, 'start', 'goal')
        return list(itemgetter(*nid_path)(self.roadmap.nodes(data='conf')))

    def _smooth_path(self,
                     component_name,
                     path,
                     obstacle_list=[],
                     otherrobot_list=[],
                     granularity=2,
                     iterations=50,
                     animation=False):
        smoothed_path = path
        for _ in range(iterations):
            if len(smoothed_path) <= 2:
                return smoothed_path
            i = random.randint(0, len(smoothed_path) - 1)
            j = random.randint(0, len(smoothed_path) - 1)
            if abs(i - j) <= 1:
                continue
            if j < i:
                i, j = j, i
            shortcut = self._extend_conf(smoothed_path[i], smoothed_path[j], granularity)
            if all(not self._is_collided(component_name=component_name,
                                         conf=conf,
                                         obstacle_list=obstacle_list,
                                         otherrobot_list=otherrobot_list)
                   for conf in shortcut):
                smoothed_path = smoothed_path[:i] + shortcut + smoothed_path[j + 1:]
            if animation:
                self.draw_wspace([self.roadmap], self.start_conf, self.goal_conf,
                                 obstacle_list, shortcut=shortcut, smoothed_path=smoothed_path)
        return smoothed_path

    def plan(self,
             component_name,
             start_conf,
             goal_conf,
             obstacle_list=[],
             otherrobot_list=[],
             ext_dist=2,
             rand_rate=70,
             max_iter=1000,
             max_time=15.0,
             smoothing_iterations=50,
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
        if self._goal_test(conf=start_conf, goal_conf=goal_conf, threshold=ext_dist):
            return [[start_conf, goal_conf], None]
        self.roadmap.add_node('start', conf=start_conf)
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
                                            ext_dist=ext_dist,
                                            goal_conf=goal_conf,
                                            obstacle_list=obstacle_list,
                                            otherrobot_list=otherrobot_list,
                                            animation=animation)
            if last_nid == 'connection':
                mapping = {'connection': 'goal'}
                self.roadmap = nx.relabel_nodes(self.roadmap, mapping)
                path = self._path_from_roadmap()
                smoothed_path = self._smooth_path(component_name=component_name,
                                                  path=path,
                                                  obstacle_list=obstacle_list,
                                                  otherrobot_list=otherrobot_list,
                                                  granularity=ext_dist,
                                                  iterations=smoothing_iterations,
                                                  animation=animation)
                return smoothed_path
        else:
            print("Reach to maximum iteration! Failed to find a path.")
            return None

    @staticmethod
    def draw_robot(plt, conf, facecolor='grey', edgecolor='grey'):
        ax = plt.gca()
        x = conf[0]
        y = conf[1]
        theta = conf[2]
        ax.add_patch(plt.Circle((x, y), .5, edgecolor=edgecolor, facecolor=facecolor))
        ax.add_patch(plt.Rectangle((x, y), .7, .1, math.degrees(theta), color='y'))
        ax.add_patch(plt.Rectangle((x, y), -.1, .1, math.degrees(theta),
                                   edgecolor=edgecolor, facecolor=facecolor))
        ax.add_patch(plt.Rectangle((x, y), .7, -.1, math.degrees(theta), color='y'))
        ax.add_patch(plt.Rectangle((x, y), -.1, -.1, math.degrees(theta),
                                   edgecolor=edgecolor, facecolor=facecolor))

    @staticmethod
    def draw_wspace(roadmap_list,
                    start_conf,
                    goal_conf,
                    obstacle_list,
                    near_rand_conf_pair=None,
                    new_conf=None,
                    shortcut=None,
                    smoothed_path=None,
                    delay_time=.02):

        plt.clf()
        ax = plt.gca()
        ax.set_aspect('equal', 'box')
        plt.grid(True)
        plt.xlim(-4.0, 17.0)
        plt.ylim(-4.0, 17.0)
        RRTDW.draw_robot(plt, start_conf, facecolor='r', edgecolor='r')
        RRTDW.draw_robot(plt, goal_conf, facecolor='g', edgecolor='g')
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
            RRTDW.draw_robot(plt, near_rand_conf_pair[0], facecolor='grey', edgecolor='g')
            RRTDW.draw_robot(plt, near_rand_conf_pair[1], facecolor='grey', edgecolor='c')
        if new_conf is not None:
            RRTDW.draw_robot(plt, new_conf, facecolor='grey', edgecolor='c')
        if smoothed_path is not None:
            plt.plot([conf[0] for conf in smoothed_path], [conf[1] for conf in smoothed_path], linewidth=7,
                     linestyle='-', color='c')
        if shortcut is not None:
            plt.plot([conf[0] for conf in shortcut], [conf[1] for conf in shortcut], linewidth=4, linestyle='--',
                     color='r')
        if not hasattr(RRTDW, 'img_counter'):
            RRTDW.img_counter = 0
        else:
            RRTDW.img_counter += 1
        # plt.savefig(str( RRT.img_counter)+'.jpg')
        if delay_time > 0:
            plt.pause(delay_time)
        # plt.waitforbuttonpress()


if __name__ == '__main__':

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
    robot = xyb.XYWBot()
    rrtdw = RRTDW(robot)
    path = rrtdw.plan(start_conf=np.array([0, 0, 0]), goal_conf=np.array([6, 9, 0]), obstacle_list=obstacle_list,
                      ext_dist=1, rand_rate=70, max_time=300, component_name='all', animation=True)
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
    rrtdw.draw_wspace([rrtdw.roadmap], rrtdw.start_conf, rrtdw.goal_conf, obstacle_list, delay_time=0)
    for conf in path:
        RRTDW.draw_robot(plt, conf, edgecolor='r')
    # pathsm = smoother.pathsmoothing(path, rrt, 30)
    # plt.plot([point[0] for point in pathsm], [point[1] for point in pathsm], '-r')
    # plt.pause(0.001)  # Need for Mac
    plt.show()

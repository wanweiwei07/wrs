import time
import math
import random
import numpy as np
import basis.robot_math as rm
import networkx as nx
import matplotlib.pyplot as plt
from operator import itemgetter


class RRT(object):

    def _decorator_keep_jnt_values(foo):
        """
        decorator function for save and restore robot_s's jnt values
        :return:
        author: weiwei
        date: 20220404
        """
        def wrapper(self, component_name, *args, **kwargs):
            jnt_values_bk = self.robot_s.get_jnt_values(component_name)
            result = foo(self, component_name, *args, **kwargs)
            self.robot_s.fk(component_name=component_name, jnt_values=jnt_values_bk)
            return result
        return wrapper

    def __init__(self, robot_s):
        self.robot_s = robot_s
        self.roadmap = nx.Graph()
        self.start_conf = None
        self.goal_conf = None

    def _is_collided(self,
                     component_name,
                     conf,
                     obstacle_list=[],
                     otherrobot_list=[]):
        """
        The function first examines if joint values of the given conf are in ranges.
        It will promptly return False if any joint value is out of range.
        Or else, it will compute fk and carry out collision checking.
        :param component_name:
        :param conf:
        :param obstacle_list:
        :param otherrobot_list:
        :return:
        author: weiwei
        date: 20220326
        """
        # self.robot_s.fk(component_name=component_name, jnt_values=conf)
        # return self.robot_s.is_collided(obstacle_list=obstacle_list, otherrobot_list=otherrobot_list)
        if self.robot_s.is_jnt_values_in_ranges(component_name=component_name, jnt_values=conf):
            self.robot_s.fk(component_name=component_name, jnt_values=conf)
            return self.robot_s.is_collided(obstacle_list=obstacle_list, otherrobot_list=otherrobot_list)
        else:
            print("The given joint angles are out of joint limits.")
            return True

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
        nodes_value_list = list(nodes_dict.values()) # attention, correspondence is not guanranteed in python
        # use the following alternative if correspondence is bad (a bit slower), 20210523, weiwei
        # # nodes_value_list = list(nodes_dict.values())
        # nodes_value_list = itemgetter(*nodes_key_list)(nodes_dict)
        # if type(nodes_value_list) == np.ndarray:
        #     nodes_value_list = [nodes_value_list]
        conf_array = np.array(nodes_value_list)
        diff_conf_array = np.linalg.norm(conf_array - new_conf, axis=1)
        min_dist_nid = np.argmin(diff_conf_array)
        return nodes_key_list[min_dist_nid]

    def _extend_conf(self, conf1, conf2, ext_dist, exact_end=True):
        """
        :param conf1:
        :param conf2:
        :param ext_dist:
        :return: a list of 1xn nparray
        """
        len, vec = rm.unit_vector(conf2 - conf1, toggle_length=True)
        # one step extension: not adopted because it is slower than full extensions, 20210523, weiwei
        # return [conf1 + ext_dist * vec]
        # switch to the following code for ful extensions
        if not exact_end:
            nval = math.ceil(len / ext_dist)
            nval = 1 if nval == 0  else nval # at least include itself
            conf_array = np.linspace(conf1, conf1 + nval * ext_dist * vec, nval)
        else:
            nval = math.floor(len / ext_dist)
            nval = 1 if nval == 0  else nval # at least include itself
            conf_array = np.linspace(conf1, conf1 + nval * ext_dist * vec, nval)
            conf_array = np.vstack((conf_array, conf2))
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
                                     new_conf, '^c')
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
            # 20210523, it seems we do not need to check line length
            # if (len(shortcut) <= (j - i)) and all(not self._is_collided(component_name=component_name,
            #                                                            conf=conf,
            #                                                            obstacle_list=obstacle_list,
            #                                                            otherrobot_list=otherrobot_list)
            #                                      for conf in shortcut):
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

    @_decorator_keep_jnt_values
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
        if not hasattr(RRT, 'img_counter'):
            RRT.img_counter = 0
        else:
            RRT.img_counter += 1
        # plt.savefig(str( RRT.img_counter)+'.jpg')
        if delay_time > 0:
            plt.pause(delay_time)
        # plt.waitforbuttonpress()


if __name__ == '__main__':
    import robot_sim.robots.xybot.xybot as xyb

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
    robot = xyb.XYBot()
    rrt = RRT(robot)
    path = rrt.plan(start_conf=np.array([0, 0]), goal_conf=np.array([6, 9]), obstacle_list=obstacle_list,
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
    rrt.draw_wspace([rrt.roadmap], rrt.start_conf, rrt.goal_conf, obstacle_list, delay_time=0)
    plt.plot([conf[0] for conf in path], [conf[1] for conf in path], linewidth=4, color='c')
    # pathsm = smoother.pathsmoothing(path, rrt, 30)
    # plt.plot([point[0] for point in pathsm], [point[1] for point in pathsm], '-r')
    # plt.pause(0.001)  # Need for Mac
    plt.show()

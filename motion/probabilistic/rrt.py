import time
import math
import random
import numpy as np
import basis.robot_math as rm
import networkx as nx
import matplotlib.pyplot as plt

class RRT(object):

    def __init__(self, robot):
        self.robot = robot.copy()
        self.roadmap = nx.Graph()
        self.start_conf = None
        self.goal_conf = None

    def _is_collided(self,
                     component_name,
                     conf,
                     obstacle_list=[],
                     otherrobot_list=[]):
        self.robot.fk(component_name=component_name, jnt_values=conf)
        return self.robot.is_collided(obstacle_list=obstacle_list, otherrobot_list=otherrobot_list)

    def _sample_conf(self, component_name, rand_rate, default_conf):
        if random.randint(0, 100) < rand_rate:
            return self.robot.rand_conf(manipulator_name=component_name)
        else:
            return default_conf

    def _get_nearest_nid(self, roadmap, new_conf):
        dist_nid_list = [[np.linalg.norm(new_conf - roadmap.nodes[nid]['conf']), nid] for nid in roadmap]
        min_dist_nid = min(dist_nid_list, key=lambda t: t[0])
        return min_dist_nid[1]

    def _extend_conf(self, conf1, conf2, ext_dist):
        """
        :param conf1:
        :param conf2:
        :param ext_dist:
        :return: a list of 1xn nparray
        """
        len, vec = rm.unit_vector(conf2 - conf1, togglelength=True)
        nval = math.ceil(len / ext_dist)
        conf_array = np.linspace(conf1, conf1 + nval * ext_dist * vec, nval)
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
        new_conf_list = self._extend_conf(roadmap.nodes[nearest_nid]['conf'], conf, ext_dist)
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
                    self.draw_wspace([roadmap], obstacle_list, [roadmap.nodes[nearest_nid]['conf'], conf], new_conf, '^c')
                # check goal
                if self._goal_test(conf=roadmap.nodes[new_nid]['conf'], goal_conf=goal_conf, threshold=ext_dist):
                    roadmap.add_node('connection', conf=goal_conf) # TODO current name -> connection
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
        conf_path = []
        for nid in nid_path:
            conf_path.append(self.roadmap.nodes[nid]['conf'])
        return conf_path

    def _smooth_path(self,
                     component_name,
                     path,
                     obstacle_list=[],
                     otherrobot_list=[],
                     granularity=2,
                     iterations=50):
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
            if (len(shortcut) < (j - i)) and all(not self._is_collided(component_name=component_name,
                                                                       conf=conf,
                                                                       obstacle_list=obstacle_list,
                                                                       otherrobot_list=otherrobot_list)
                                                 for conf in shortcut):
                smoothed_path = smoothed_path[:i + 1] + shortcut + smoothed_path[j + 1:]
        return smoothed_path

    def plan(self,
             component_name,
             start_conf,
             goal_conf,
             obstacle_list=[],
             otherrobot_list=[],
             ext_dist=2,
             rand_rate=70,
             maxiter=1000,
             maxtime=15.0,
             animation=False):
        """
        :return: [path, all_sampled_confs]
        author: weiwei
        date: 20201226
        """
        self.roadmap.clear()
        self.start_conf = start_conf
        self.goal_conf = goal_conf
        # check seed_jnt_values and goal_conf
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
        for _ in range(maxiter):
            toc = time.time()
            if maxtime > 0.0:
                if toc - tic > maxtime:
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
                                            animation=True)
            if last_nid == 'connection':
                mapping = {'connection':'goal'}
                self.roadmap = nx.relabel_nodes(self.roadmap, mapping)
                path = self._path_from_roadmap()
                smoothed_path = self._smooth_path(component_name=component_name,
                                                  path=path,
                                                  obstacle_list=obstacle_list,
                                                  otherrobot_list=otherrobot_list,
                                                  granularity=ext_dist,
                                                  iterations=100)
                return smoothed_path
        else:
            print("Reach to maximum iteration! Failed to find a path.")
            return None

    @staticmethod
    def draw_wspace(roadmap_list, obstacle_list, near_rand_conf_pair=None, new_conf=None, new_conf_mark='^r'):
        """
        Draw Graph
        """
        plt.clf()
        ax = plt.gca()
        ax.set_aspect('equal', 'box')
        plt.grid(True)
        plt.xlim(-4.0, 17.0)
        plt.ylim(-4.0, 17.0)
        for (point, size) in obstacle_list:
            ax.add_patch(plt.Circle((point[0], point[1]), size / 2.0, color='k'))
        colors = 'bgrcmykw'
        for i, roadmap in enumerate(roadmap_list):
            for (u, v) in roadmap.edges:
                plt.plot(roadmap.nodes[u]['conf'][0], roadmap.nodes[u]['conf'][1], 'o'+colors[i])
                plt.plot(roadmap.nodes[v]['conf'][0], roadmap.nodes[v]['conf'][1], 'o'+colors[i])
                plt.plot([roadmap.nodes[u]['conf'][0], roadmap.nodes[v]['conf'][0]],
                         [roadmap.nodes[u]['conf'][1], roadmap.nodes[v]['conf'][1]], '-'+colors[i])
        if near_rand_conf_pair is not None:
            plt.plot([near_rand_conf_pair[0][0], near_rand_conf_pair[1][0]],
                     [near_rand_conf_pair[0][1], near_rand_conf_pair[1][1]], "--k")
        if new_conf is not None:
            plt.plot(new_conf[0], new_conf[1], new_conf_mark)
        # plt.plot(planner.seed_jnt_values[0], planner.seed_jnt_values[1], "xr")
        # plt.plot(planner.goal_conf[0], planner.goal_conf[1], "xm")
        plt.pause(.02)


if __name__ == '__main__':
    import robotsim._kinematics.jlchain as jl
    import robotsim.robots.robot_interface as ri


    class XYBot(ri.RobotInterface):

        def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name='XYBot'):
            super().__init__(pos=pos, rotmat=rotmat, name=name)
            self.jlc = jl.JLChain(homeconf=np.zeros(2), name='XYBot')
            self.jlc.jnts[1]['type'] = 'prismatic'
            self.jlc.jnts[1]['loc_motionax'] = np.array([1, 0, 0])
            self.jlc.jnts[1]['loc_pos'] = np.zeros(3)
            self.jlc.jnts[1]['motion_rng'] = [-2.0, 15.0]
            self.jlc.jnts[2]['type'] = 'prismatic'
            self.jlc.jnts[2]['loc_motionax'] = np.array([0, 1, 0])
            self.jlc.jnts[2]['loc_pos'] = np.zeros(3)
            self.jlc.jnts[2]['motion_rng'] = [-2.0, 15.0]
            self.jlc.reinitialize()

        def fk(self, component_name='all', jnt_values=np.zeros(2)):
            if component_name != 'all':
                raise ValueError("Only support component_name == 'all'!")
            self.jlc.fk(jnt_values)

        def rand_conf(self, component_name='all'):
            if component_name != 'all':
                raise ValueError("Only support component_name == 'all'!")
            return self.jlc.rand_conf()

        def get_jntvalues(self, component_name='all'):
            if component_name != 'all':
                raise ValueError("Only support component_name == 'all'!")
            return self.jlc.get_jnt_values()

        def is_collided(self, obstacle_list=[], otherrobot_list=[]):
            for (obpos, size) in obstacle_list:
                dist = np.linalg.norm(np.asarray(obpos) - self.get_jntvalues())
                if dist <= size / 2.0:
                    return True  # collision
            return False  # safe


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
    robot = XYBot()
    rrt = RRT(robot)
    path = rrt.plan(start_conf=np.array([0, 0]), goal_conf=np.array([5, 10]), obstacle_list=obstacle_list,
                    ext_dist=1, rand_rate=70, maxtime=300, component_name='all', animation=True)
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
    plt.plot([conf[0] for conf in path], [conf[1] for conf in path], '-k')
    # pathsm = smoother.pathsmoothing(path, rrt, 30)
    # plt.plot([point[0] for point in pathsm], [point[1] for point in pathsm], '-r')
    # plt.pause(0.001)  # Need for Mac
    plt.show()

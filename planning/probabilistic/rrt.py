import time
import math
import random
import numpy as np
import basis.robotmath as rm
import networkx as nx


class RRT(object):

    def __init__(self, robot):
        self.robot = robot.copy()
        self.roadmap = nx.DiGraph()
        self.start = None
        self.goal = None

    def _is_collided(self, jlc_name, conf, obstacle_list=[], otherrobot_list=[]):
        if jlc_name is None:
            self.robot.fk(conf)
        else:
            self.robot.fk(conf, jlc_name=jlc_name)
        return self.robot.is_collided(obstacle_list=obstacle_list, otherrobot_list=otherrobot_list)

    def _sample_conf(self, rand_rate, default_conf):
        if random.randint(0, 100) < rand_rate:
            return self.robot.rand_conf()
        else:
            return default_conf

    def _get_nearest_nid(self, new_conf):
        dist_nid_list = [[np.linalg.norm(new_conf - self.roadmap.node[nid]['conf']), nid] for nid in self.roadmap]
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

    def _goal_test(self, conf, goal, threshold):
        dist = np.linalg.norm(conf - goal)
        if dist <= threshold:
            print("Goal reached!")
            return True
        else:
            return False

    def _path_from_roadmap(self):
        nid_path = nx.shortest_path(self.roadmap, 'start', 'goal')
        conf_path = []
        for nid in nid_path:
            conf_path.append(self.roadmap.node[nid]['conf'])
        return conf_path

    def _smooth_path(self,
                     path,
                     jlc_name=None,
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
            if (len(shortcut) < (j - i)) and all(not self._is_collided(jlc_name=jlc_name,
                                                                       conf=conf,
                                                                       obstacle_list=obstacle_list,
                                                                       otherrobot_list=otherrobot_list)
                                                 for conf in shortcut):
                smoothed_path = smoothed_path[:i + 1] + shortcut + smoothed_path[j + 1:]
        return smoothed_path

    def plan(self, start, goal, obstacle_list=[], otherrobot_list=[], ext_dist=2, rand_rate=70,
             maxiter=1000, maxtime=15.0, animation=False, jlc_name=None):
        """
        :return: [path, all_sampled_confs]
        author: weiwei
        date: 20201226
        """
        self.roadmap.clear()
        self.start = start
        self.goal = goal
        # check start and goal
        if self._is_collided(jlc_name, start, obstacle_list, otherrobot_list):
            print("The start robot configuration is in collision!")
            return [None, None]
        if self._is_collided(jlc_name, goal, obstacle_list, otherrobot_list):
            print("The goal robot configuration is in collision!")
            return [None, None]
        if self._goal_test(conf=start, goal=goal, threshold=ext_dist):
            return [[start, goal], None]
        # all sampled confs: [node, is_collided]
        all_sampled_confs = []
        self.roadmap.add_node('start', conf=start)
        iter_count = 0
        tic = time.time()
        while True:
            toc = time.time()
            if maxtime > 0.0:
                if toc - tic > maxtime:
                    print("Too much planning time! Failed to find a path.")
                    return [None, None]
            if iter_count > maxiter:
                print("Reach to maximum iteration! Failed to find a path.")
                return [None, None]
            # Random Sampling
            rand_conf = self._sample_conf(rand_rate=rand_rate, default_conf=goal)
            # Find nearest node
            nearest_nid = self._get_nearest_nid(rand_conf)
            new_conf_list = self._extend_conf(self.roadmap.node[nearest_nid]['conf'], rand_conf, ext_dist)
            for new_conf in new_conf_list:
                if self._is_collided(jlc_name, new_conf, obstacle_list, otherrobot_list):
                    if animation:
                        drawwspace(self, obstacle_list, rand_conf, new_conf, '^b')
                    break
                else:
                    new_nid = random.randint(0, 1e16)
                    self.roadmap.add_node(new_nid, conf=new_conf)
                    self.roadmap.add_edge(nearest_nid, new_nid)
                    nearest_nid = new_nid
                    # all_sampled_confs.append([new_node.point, False])
                    if animation:
                        drawwspace(self, obstacle_list, rand_conf, new_conf, '^c')
                    # check goal
                    if self._goal_test(conf=self.roadmap.node[new_nid]['conf'], goal=goal, threshold=ext_dist):
                        self.roadmap.add_node('goal', conf=goal)
                        self.roadmap.add_edge(new_nid, 'goal')
                        path = self._path_from_roadmap()
                        smoothed_path = self._smooth_path(path,
                                                          jlc_name=jlc_name,
                                                          obstacle_list=obstacle_list,
                                                          otherrobot_list=otherrobot_list,
                                                          granularity=ext_dist,
                                                          iterations=100)
                        return smoothed_path
            iter_count += 1


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import robotsim._kinematics.jlchain as jl


    def drawwspace(planner, obstacle_list, rand_conf=None, new_conf=None, new_conf_mark='^r'):
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
        for (u, v) in planner.roadmap.edges:
            print(u, v)
            plt.plot(planner.roadmap.node[u]['conf'][0], planner.roadmap.node[u]['conf'][1], 'og')
            plt.plot(planner.roadmap.node[v]['conf'][0], planner.roadmap.node[v]['conf'][1], 'og')
            plt.plot([planner.roadmap.node[u]['conf'][0], planner.roadmap.node[v]['conf'][0]],
                     [planner.roadmap.node[u]['conf'][1], planner.roadmap.node[v]['conf'][1]], '-g')
        if rand_conf is not None:
            plt.plot(rand_conf[0], rand_conf[1], "^k")
        if new_conf is not None:
            plt.plot(new_conf[0], new_conf[1], new_conf_mark)
        plt.plot(planner.start[0], planner.start[1], "xr")
        plt.plot(planner.goal[0], planner.goal[1], "xm")
        plt.pause(.101)


    class XYBot(jl.JLChain):

        def __init__(self):
            super().__init__(homeconf=np.zeros(2), name='XYBot')
            self.jnts[1]['type'] = 'prismatic'
            self.jnts[1]['loc_motionax'] = np.array([1, 0, 0])
            self.jnts[1]['loc_pos'] = np.zeros(3)
            self.jnts[1]['motion_rng'] = [-2.0, 15.0]
            self.jnts[2]['type'] = 'prismatic'
            self.jnts[2]['loc_motionax'] = np.array([0, 1, 0])
            self.jnts[2]['loc_pos'] = np.zeros(3)
            self.jnts[2]['motion_rng'] = [-2.0, 15.0]
            self.reinitialize()

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
    path = rrt.plan(start=np.array([0, 0]), goal=np.array([5, 10]), obstacle_list=obstacle_list,
                    ext_dist=1, rand_rate=70, maxtime=300, jlc_name=None, animation=True)
    # plt.show()
    # nx.draw(rrt.roadmap, with_labels=True, font_weight='bold')
    # plt.show()
    # import time
    # total_t = 0
    # for i in range(1):
    #     tic = time.time()
    #     path, sampledpoints = rrt.planning(obstaclelist=obstaclelist, animation=True)
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

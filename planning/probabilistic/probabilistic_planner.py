import math


class ProbabilisticPlannerInterface(object):

    def __init__(self, robot):
        self.robot = robot

    def planning(self, start, goal):
        raise NotImplementedError
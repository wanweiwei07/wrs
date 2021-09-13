import enum
import numpy as np


class State(object):
    def __init__(self, row=-1, column=-1):
        self.row = row
        self.column = column

    def __repr__(self):
        return f"<State: [{self.row} {self.column}]"

    def clone(self):
        return State(self.row, self.column)

    def __hash__(self):
        return hash((self.row, self.column))

    def __eq__(self, other):
        return self.row == other.row and self.column == other.column


class Action(enum.Enum):
    UP = 1
    DOWN = -1
    LEFT = 2
    RIGHT = -2


class Environment(object):

    def __init__(self, grid, prob=.8):
        self.grid = grid
        self.agent_state = State()
        self.move_prob = prob
        self.default_reward = -0.4
        self.reset()

    @property
    def nrow(self):
        return len(self.grid)

    @property
    def ncol(self):
        return len(self.grid[0])

    @property
    def actions(self):
        return (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT)

    @property
    def states(self):
        states = []
        for row in range(self.nrow):
            for column in range(self.ncol):
                if self.grid[row][column] != 9:
                    states.append(State(row, column))
        return states

    def can_action_at(self, state):
        return True if self.grid[state.row][state.column] == 0 else False

    def _move(self, state, action):
        if not self.can_action_at(state):
            raise Exception("Cannot move from here!")
        next_state = state.clone()
        if action == Action.UP:
            next_state.row -= 1
        elif action == Action.DOWN:
            next_state.row += 1
        elif action == Action.LEFT:
            next_state.column -= 1
        elif action == Action.RIGHT:
            next_state.column += 1
        if next_state.row < 0 or next_state.row >= self.nrow:
            next_state = state
        if next_state.column < 0 or next_state.column >= self.ncol:
            next_state = state
        if self.grid[next_state.row][next_state.column] == 9:
            next_state = state
        return next_state

    def transit_func(self, state, action):
        transition_probs = {}
        if not self.can_action_at(state):
            return transition_probs
        opposite_direction = Action(action.value * -1)
        for a in self.actions:
            prob = 0
            if a == action:
                prob = self.move_prob
            elif a != opposite_direction:
                prob = (1 - self.move_prob) / 2
            next_state = self._move(state, a)
            if next_state not in transition_probs:
                transition_probs[next_state] = prob
            else:
                transition_probs[next_state] += prob
        return transition_probs

    def reward_func(self, state):
        reward = self.default_reward
        done = False
        if self.grid[state.row][state.column] == 1:
            reward = 1
            done = True
        elif self.grid[state.row][state.column] == -1:
            reward = -1
            done = True
        return reward, done

    def transit(self, state, action):
        transition_probs = self.transit_func(state, action)
        if len(transition_probs) == 0:
            return None, None, True
        next_state = np.random.choice(list(transition_probs.keys()), p=list(transition_probs.values()))
        reward, done = self.reward_func(next_state)
        return next_state, reward, done

    def step(self, action):
        next_state, reward, done = self.transit(self.agent_state, action)
        if next_state is not None:
            self.agent_state = next_state
        return next_state, reward, done

    def reset(self):
        self.agent_state = State(self.nrow - 1, 0)
        return self.agent_state


if __name__ == '__main__':
    import random
    import sys


    class Agent(object):
        def __init__(self, env):
            self.actions = env.actions

        def policy(self, state):
            return random.choice(self.actions)


    class ValueIteration(object):
        def __init__(self, env):
            self.env = env
            self.V = {}

        def plan(self, gamma=.8):
            self.V = {}
            for s in self.env.states:
                self.V[s] = 0
            while True:
                delta = 0
                for s in self.V:
                    if not self.env.can_action_at(s):
                        continue
                    expected_values = []
                    for a in self.env.actions:
                        transition_probs = self.env.transit_func(s, a)
                        v = 0
                        for next_state in transition_probs:
                            v += transition_probs[next_state] * (
                                        self.env.reward_func(next_state)[0] + gamma * self.V[next_state])
                        expected_values.append(v)
                    max_value = max(expected_values)
                    delta = max(delta, abs(max_value - self.V[s]))
                    self.V[s] = max_value
                if delta < 1e-4:
                    break


    grid = [[0, 0, 0, 1],
            [0, 9, 0, -1],
            [0, 0, 0, 0]]
    env = Environment(grid)
    agent = Agent(env)
    learner = ValueIteration(env)
    learner.plan()
    values_array = np.zeros_like(grid).astype(np.float32)
    for s in learner.V:
        print(values_array[s.row, s.column], learner.V[s])
        values_array[s.row, s.column] = learner.V[s]
        print(values_array[s.row, s.column], learner.V[s])
    print(values_array)
    # for i in range(10):
    #     state = learner.env.reset()
    #     total_reward = 0
    #     done = False
    #
    #     while not done:
    #         action = agent.policy(state)
    #         next_state, reward, done = env.step(action)
    #         total_reward += reward
    #         state = next_state
    #     print(f"Episode {i}: Agent gets {total_reward} reward.")

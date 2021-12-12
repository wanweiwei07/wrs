import numpy as np
from file_sys import load_pickle
from pathlib import Path
from env_meta.env_meta import (get_fillable_movable,
                               isdone,
                               get_random_states,
                               get_random_goal_pattern)


def get_feasible_action_set(state, rack_size):
    fillable, movable = get_fillable_movable(state)
    fillable_idx = np.where(fillable.ravel() == 1)[0]
    movable_idx = np.where(movable.ravel() == 1)[0]
    if len(fillable_idx) == 0 or len(movable_idx) == 0:
        # raise Exception("ERROR IN GET_FEASIBLE_ACTION_SET")
        return np.array([0])  # must be done
    fillable_movable_idx_comb = np.array(np.meshgrid(fillable_idx, movable_idx)).T.reshape(-1, 2)
    return fillable_movable_idx_comb[:, 0] * np.prod(rack_size) + fillable_movable_idx_comb[:, 1]


GOAL = np.array([
    [1, 1, 1, 1, 0, 0, 2, 2, 2, 2],
    [1, 1, 1, 1, 0, 0, 2, 2, 2, 2],
    [1, 1, 1, 1, 0, 0, 2, 2, 2, 2],
    [1, 1, 1, 1, 0, 0, 2, 2, 2, 2],
    [1, 1, 1, 1, 0, 0, 2, 2, 2, 2],
])


class ENV:
    def __init__(self, rack_size=(5, 10), num_classes=5, observation_space_dim=10, action_space_dim=10):
        self.rack_size = rack_size
        self.num_classes = num_classes
        self.observation_space_dim = observation_space_dim
        self.action_space_dim = action_space_dim
        self.state = np.zeros(observation_space_dim)
        self.goal_pattern = None
        self.rack_state_history = []
        self.reward_history = []

    def reset_state_goal(self, initstate, goal_pattern):
        self.goal_pattern = goal_pattern
        self.state = initstate
        self.rack_state_history = []
        self.reward_history = []
        return self.state

    def reset(self):
        # set random goal
        # generate random goal patterns
        # goal_pattern = np.random.randint(1, self.num_classes + 1, size=self.rack_shape)
        goal_pattern = GOAL.copy()
        initstate = get_random_states(self.rack_size, goal_pattern, )
        print("--------------")
        # print("goal pattern is:")
        # print(goal_pattern)
        print("init pattern is:")
        print(initstate)
        print("--------------")
        # goal_pattern_multilayer = seperate_matrix_layer(goal_pattern.copy(), self.num_classes)
        # initstate_multilayer = seperate_matrix_layer(init_state.copy(), self.num_classes)
        return self.reset_state_goal(initstate=initstate, goal_pattern=goal_pattern)

    def sample_action_space(self, state):
        return np.random.choice(get_feasible_action_set(state, rack_size=self.rack_size))

    def _expr_action(self, actionidx):
        rack_size = self.rack_size
        selected_obj = actionidx % np.prod(rack_size)
        selected_obj_row = selected_obj // rack_size[1]
        selected_obj_column = selected_obj % rack_size[1]
        goal_pos = actionidx // np.prod(rack_size)
        goal_pos_row = goal_pos // rack_size[1]
        goal_pos_column = goal_pos % rack_size[1]
        return (selected_obj_row, selected_obj_column), (goal_pos_row, goal_pos_column)

    def _get_reward(self, is_finished, old_state, new_state, goal_pattern):
        if is_finished:
            return 50
        move_map = new_state - old_state
        move_to_idx = np.where(move_map > 0)
        move_from_idx = np.where(move_map < 0)
        is_move_to_pattern = (goal_pattern[move_to_idx] == new_state[move_to_idx]).item()
        is_in_pattern = (goal_pattern[move_from_idx] == old_state[move_from_idx]).item()
        if is_move_to_pattern and not is_in_pattern:
            return 1
        return 0

    def _act(self, obj_id, goal_id):
        # current state
        curr_state = self.state
        goal_pattern = self.goal_pattern
        self.rack_state_history.append(curr_state.copy())
        # new state, update to new state
        ## --------------------
        nxt_state = curr_state.copy()
        nxt_state[goal_id] = curr_state[obj_id]
        nxt_state[obj_id] = 0
        # update new state
        self.state = nxt_state
        ## --------------------
        # check if the new state is illegal state
        nxt_fillable, nxt_movable = get_fillable_movable(nxt_state)
        if np.sum(nxt_fillable) == 0 or np.sum(nxt_movable) == 0:
            reward = -1
            self.reward_history.append(reward)
            return self.state, reward, True
        is_finished = isdone(nxt_state.copy(), goal_pattern)
        # get reward of the action
        reward = self._get_reward(is_finished, curr_state.copy(), nxt_state.copy(), goal_pattern)
        self.reward_history.append(reward)
        # return reward, isdone
        return self.state, reward, is_finished

    def step(self, action):
        obj_id, goal_id = self._expr_action(action)
        new_obs, reward, is_done = self._act(obj_id, goal_id)
        return new_obs, reward, is_done, {
            "reward_history": self.reward_history
        }

    def gen_Astar_solution(self, max_iter_cnt=500):
        rack_size = self.rack_size
        goal_pattern = self.goal_pattern.copy()
        elearray = self.state.copy()
        tp = TubePuzzle(elearray.copy())
        tp.goalpattern = goal_pattern.copy()
        try:
            path = tp.atarSearch(max_iter_cnt=max_iter_cnt)
        except:
            return None
        if path is None:
            return None
        action_seq = []
        for idx, p in enumerate(path):
            if idx == 0:
                continue
            last_old_state = path[idx - 1]
            new_state = path[idx]
            move_map = new_state.grid - last_old_state.grid
            move_to_idx = np.where(move_map > 0)
            move_from_idx = np.where(move_map < 0)
            moveidx, fillidx = np.asarray(move_from_idx).T[0], np.asarray(move_to_idx).T[0]
            moveidx_1d = moveidx[0] * rack_size[1] + moveidx[1]
            fillidx_1d = fillidx[0] * rack_size[1] + fillidx[1]
            action_encode = fillidx_1d * np.prod(rack_size) + moveidx_1d
            action_seq.append(action_encode)
        return action_seq


class ENV_data(ENV):
    def __init__(self, data_path, rack_size=(5, 10), num_classes=4, observation_space_dim=10, action_space_dim=10):
        super(ENV_data, self).__init__(rack_size, num_classes, observation_space_dim, action_space_dim)
        self.data_path = Path(data_path)
        self.eposide_buffer = []
        self.eposide_buffer_cnt = 0
        self.current_eposide = None
        self.action_list = None
        for data_file_name in self.data_path.glob("*"):
            self.eposide_buffer = self.eposide_buffer + load_pickle(data_file_name)

    def reset(self):
        self.current_eposide = self.eposide_buffer[self.eposide_buffer_cnt]
        self.action_list = self.current_eposide['action_seq']
        self.eposide_buffer_cnt += 1
        self.eposide_buffer_cnt = self.eposide_buffer_cnt % len(self.eposide_buffer)
        return self.reset_state_goal(self.current_eposide['start'], self.current_eposide['goal'])

    def sample_action_space(self, state):
        if self.action_list is None or len(self.action_list) == 0:
            raise Exception("Error")
        rack_size = self.rack_size
        moveidx, fillidx = self.action_list.pop(0)
        moveidx_1d = moveidx[0] * rack_size[1] + moveidx[1]
        fillidx_1d = fillidx[0] * rack_size[1] + fillidx[1]
        action_random = fillidx_1d * np.prod(rack_size) + moveidx_1d
        return action_random


if __name__ == "__main__":
    old_state = np.array([
        [2, 0, 1, 2, 2, 1, 2, 0, 2, 1],
        [0, 1, 2, 2, 0, 2, 1, 2, 1, 1],
        [0, 1, 1, 1, 1, 0, 0, 2, 1, 0],
        [1, 0, 0, 2, 1, 0, 2, 2, 0, 0],
        [2, 1, 1, 1, 1, 2, 1, 2, 0, 2]
    ])

    new_state = np.array([[0, 0, 1, 2, 2, 1, 2, 0, 2, 1],
                      [0, 1, 2, 2, 0, 2, 1, 2, 1, 1],
                      [0, 1, 1, 1, 1, 0, 0, 2, 1, 0],
                      [1, 0, 0, 2, 1, 0, 2, 2, 0, 2],
                      [2, 1, 1, 1, 1, 2, 1, 2, 0, 2]])

    goal_pattern = GOAL

    move_map = new_state - old_state
    print(move_map)
    move_to_idx = np.where(move_map > 0)
    move_from_idx = np.where(move_map < 0)
    is_move_to_pattern = (goal_pattern[move_to_idx] == new_state[move_to_idx]).item()
    print(goal_pattern[move_to_idx])
    print(new_state[move_to_idx])
    print(move_to_idx)
    is_in_pattern = (goal_pattern[move_from_idx] == old_state[move_from_idx]).item()
    print(goal_pattern[move_to_idx])
    print(old_state[move_from_idx])
    print(is_move_to_pattern)
    print(is_in_pattern)
    exit(0)

    rack_size = (5, 10)
    num_classes = 2
    obs_dim = (num_classes * 2, *rack_size)
    act_dim = np.prod(rack_size) ** 2
    env = ENV(rack_size=rack_size, num_classes=num_classes, observation_space_dim=obs_dim, action_space_dim=act_dim)
    obs = env.reset()
    print("obs is\n", obs)
    action = env.sample_action_space(obs)
    new_obs, reward, is_done, _ = env.step(action)
    print(new_obs)
    print(reward)
    new_obs = obs
    # print(f"action is {env._expr_action(action)}")
    # new_obs, reward, is_done, _ = env.step(action)
    # print(env.rack_state_history)
    # print(new_obs)
    # print(reward)
    # print(is_done)
    # print(new_obs)

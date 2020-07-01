import os
import numpy as np
import pickle
from scipy.sparse import csr_matrix, lil_matrix
from matplotlib import pyplot as plt
import cv2
from scipy import interpolate

from constants import *
import discrete_platformer
import value_iteration
import unsafe_grid_world


tableau10 = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189),
             (140, 86, 75), (227, 119, 194), (127, 127, 127), (188, 189, 34), (23, 190, 207)]
for i in range(len(tableau10)):
    r, g, b = tableau10[i]
    tableau10[i] = (r / 255., g / 255., b / 255.)

agent_labels_dict = {
    'ase-agent': 'ASE',
    'undirected-ase-agent': 'Undirected ASE',
    'mbie-agent': 'MBIE (unsafe)',
    'safe-rmax-agent': 'Safe R-Max',
    'rmax-agent': 'R-Max (unsafe)',
    'safe-egreedy-agent': r'Safe $\epsilon$-greedy',
    'egreedy-agent': r'$\epsilon$-greedy (unsafe)'
}


def plot_num_sub_optimal(agent_names, experiment):
    experiment_dir = os.path.join(os.getcwd(), 'results', 'experiment%02d' % experiment, agent_names[0], 'trial%02d' % 1)
    state_history_fn = os.path.join(experiment_dir, 'state_history.pkl')
    param_dict_fn = os.path.join(experiment_dir, 'param_dict.pkl')
    state_history = pickle.load(open(state_history_fn, 'rb'))
    param_dict = pickle.load(open(param_dict_fn, 'rb'))
    num_trials = param_dict['num_trials']
    max_num_steps = 30000
    plt.figure(figsize=(10, 6))
    
    environment_name = param_dict['environment']
    if environment_name == 'discrete_platformer':
        env, _ = discrete_platformer.env_from_file(param_dict['level'])
    elif environment_name == 'unsafe_grid_world':
        env = unsafe_grid_world.make_grid_world_from_file(param_dict['level'], slip_prob=0.6, gui=True)
    else:
        ValueError('Unrecognized Environment: %s' % environment_name)
        return
    
    all_states = env.get_all_states()
    state_to_id = dict()
    for s_id, state in enumerate(all_states):
        state_to_id[state] = s_id
    all_actions = env.get_actions()
    action_to_id = dict()
    for i, a in enumerate(all_actions):
        action_to_id[a] = i
    num_states = len(all_states)
    num_actions = len(all_actions)
    true_rewards = np.zeros([num_states, num_actions], dtype=DTYPE)
    terminal_states = np.zeros([num_states], dtype=DTYPE)
    true_transitions = lil_matrix((num_states * num_actions, num_states), dtype=DTYPE)
    for s, state in enumerate(all_states):
        for a, action in enumerate(all_actions):
            true_rewards[s, a] = env.reward_function(state, action)
            if state.is_terminal():
                terminal_states[s] = 1
        
            sa = np.ravel_multi_index([s, a], [num_states, num_actions])
            next_states, p = env.transition_function(state, action)
            for i, next_state in enumerate(next_states):
                if next_state in state_to_id:
                    sp = state_to_id[next_state]
                    true_transitions[sa, sp] = p[i]

    true_transitions = true_transitions.tocsr()

    optimal_values, optimal_qs = value_iteration.value_iteration(true_transitions, true_rewards, terminal_states,
                                                    horizon=param_dict['vi_horizon'], gamma=param_dict['gamma'],
                                                    action_mask=true_rewards >= 0, q_default=-np.infty,
                                                    use_sparse_matrices=True)
    # optimal_values, optimal_qs = value_iteration.value_iteration(true_transitions, true_rewards, terminal_states,
    #                                                              horizon=param_dict['vi_horizon'],
    #                                                              gamma=param_dict['gamma'],
    #                                                              use_sparse_matrices=True)
    epsilon = 0.01
    eps_opt_actions = optimal_qs >= (optimal_values - epsilon)[:, np.newaxis]
    num_steps = None

    for agent_i, agent_name in enumerate(agent_names):
        all_num_sub_opt = []
        num_unsafe_states = np.zeros([num_trials])
        for t in range(num_trials):
            experiment_dir = os.path.join(os.getcwd(), 'results', 'experiment%02d' % experiment, agent_name, 'trial%02d' % (t+1))
            param_dict_fn = os.path.join(experiment_dir, 'param_dict.pkl')
            state_history_fn = os.path.join(experiment_dir, 'state_history.pkl')
            action_history_fn = os.path.join(experiment_dir, 'action_history.pkl')
            reward_history_fn = os.path.join(experiment_dir, 'reward_history.pkl')
    
            param_dict = pickle.load(open(param_dict_fn, 'rb'))
            state_history = pickle.load(open(state_history_fn, 'rb'))
            action_history = pickle.load(open(action_history_fn, 'rb'))
            reward_history = pickle.load(open(reward_history_fn, 'rb'))

            n = 0
            r = 0
            num_sub_opt = []
            rs = []
            for i in range(len(action_history)):
                if len(num_sub_opt) >= max_num_steps:
                    break
                
                if r != 0:
                    r = 0
                    continue
                
                r = reward_history[i]
                rs.append(r)

                state = state_history[i]
                action = action_history[i]
                s = state_to_id[state]
                a = action_to_id[action]
                
                if not eps_opt_actions[s, a]:
                    n += 1
                num_sub_opt.append(n)
            all_num_sub_opt.append(num_sub_opt)
            num_unsafe_states[t] = (np.array(rs) < 0).sum()
        
        if num_steps is not None:
            assert num_steps == len(all_num_sub_opt[0])
        num_steps = len(all_num_sub_opt[0])
        for t in range(num_trials):
            assert num_steps == len(all_num_sub_opt[1])
        num_sub_opt = np.array(all_num_sub_opt)
        
        print('Average unsafe states for agent %s: %0.2f' % (agent_name, num_unsafe_states.mean()))
        xs = np.arange(num_steps)/1000
        num_sub_opt = num_sub_opt / 1000
        zorder = len(agent_names) - agent_i
        plt.plot(xs, num_sub_opt.mean(axis=0), color=tableau10[agent_i], label=agent_labels_dict[agent_name], zorder=zorder)
        plt.fill_between(xs, num_sub_opt.max(axis=0), num_sub_opt.min(axis=0), color=tableau10[agent_i], alpha=0.5, zorder=zorder)

    plt.legend(fontsize=18)
    plt.xlabel('Time steps (thousands)', fontsize=18)
    plt.ylabel(r'$\epsilon$-sub-optimal steps (thousands)', fontsize=18)

    experiment_dir = os.path.join(os.getcwd(), 'results', 'experiment%02d' % experiment)
    plot_fn = os.path.join(experiment_dir, 'num_sub_opt_plot.pdf')
    plt.savefig(plot_fn)
    
    plt.show()


def plot_heat_map(agent_names, experiment):
    experiment_dir = os.path.join(os.getcwd(), 'results', 'experiment%02d' % experiment, agent_names[0], 'trial%02d' % 1)
    state_history_fn = os.path.join(experiment_dir, 'state_history.pkl')
    param_dict_fn = os.path.join(experiment_dir, 'param_dict.pkl')
    state_history = pickle.load(open(state_history_fn, 'rb'))
    param_dict = pickle.load(open(param_dict_fn, 'rb'))
    num_trials = param_dict['num_trials']
    
    env, _ = discrete_platformer.env_from_file(param_dict['level'])
    states = env.get_all_states()
    xs = [state.x for state in states]
    ys = [state.y for state in states]
    min_x = min(xs)
    max_x = max(xs)
    min_y = 0
    max_y = max(ys)

    for agent_name in agent_names:
        plt.figure(figsize=(40, 10))
        
        map_image = env.gui.map_image(min_x, max_x, min_y, max_y)
        plt.imshow(map_image, alpha=1)
        plt.xlim([5 * discrete_platformer.TILE_WIDTH, map_image.shape[1] - 5 * discrete_platformer.TILE_WIDTH])
        
        heat_map = np.zeros([max_x - min_x + 1, max_y - min_y + 1])
        xs = []
        ys = []
        in_air = []
        rs = []
        safe_trajectory = []
        
        t = 0
        
        experiment_dir = os.path.join(os.getcwd(), 'results', 'experiment%02d' % experiment, agent_name,
                                      'trial%02d' % (t + 1))
        param_dict_fn = os.path.join(experiment_dir, 'param_dict.pkl')
        state_history_fn = os.path.join(experiment_dir, 'state_history.pkl')
        reward_history_fn = os.path.join(experiment_dir, 'reward_history.pkl')
    
        param_dict = pickle.load(open(param_dict_fn, 'rb'))
        state_history = pickle.load(open(state_history_fn, 'rb'))
        reward_history = pickle.load(open(reward_history_fn, 'rb'))
        
        for i in range(len(state_history)):
            state = state_history[i]
            x = state.x - min_x
            y = max_y - (state.y - min_y)
            in_air.append(1 if state.y > 1 else 0)
            heat_map[x, y] += 1
            xs.append((x + 1.5) * discrete_platformer.TILE_WIDTH)
            ys.append((y + 1) * discrete_platformer.TILE_HEIGHT)
            if i < len(reward_history):
                r = reward_history[i]
                rs.append(r)
                if r < 0:
                    safe_trajectory.append(0)
                elif r > 0:
                    safe_trajectory.append(1)
        safe_trajectory.append(0)
        
        trajectory_index = 0
        j = 0
        for i in range(len(state_history)):
            if i > 0 and rs[i - 1] != 0:
                trajectory_index += 1
            if (i > 0 and rs[i - 1] != 0 and i > j) \
                    or (i > 0 and in_air[i-1] and not in_air[i]) \
                    or (i < len(state_history) - 1 and not in_air[i] and in_air[i+1]):
                color = 'r' if rs[i - 1] < 0 else 'b'
                alpha = 0.1 if rs[i - 1] < 0 else 0.02
                if i > 0 and in_air[i-1] and xs[j+1] != xs[i] and i - j >= 3:
                    tck, u = interpolate.splprep([xs[j:i + 1], ys[j:i + 1]], s=0)
                    unew = np.arange(0, 1.01, 0.01)
                    new_points = interpolate.splev(unew, tck)
                    plt.plot(new_points[0], new_points[1], alpha=alpha, lw=5, solid_capstyle="round", c=color)
                else:
                    length_of_dangerous_traj = 2
                    if rs[i - 1] < 0 and i - length_of_dangerous_traj > j:
                        j_prime = i - length_of_dangerous_traj
                        plt.plot(xs[j:j_prime+1], ys[j:j_prime+1], alpha=0.02, lw=5, solid_capstyle="round", c='b')
                        j = j_prime
                    plt.plot(xs[j:i+1], ys[j:i+1], alpha=alpha, lw=5, solid_capstyle="round", c=color)

                if rs[i] != 0:
                    j = i + 1
                else:
                    j = i
            
        # heat_map /= num_trials
        # heat_map = heat_map.transpose([1, 0])
        # heat_map = cv2.resize(heat_map, (map_image.shape[1], map_image.shape[0]), interpolation=cv2.INTER_NEAREST)
        # plt.imshow(heat_map, alpha=1.0)
        
        # plt.imshow(map_image, alpha=0.5)
    
        plt.tick_params(axis='both', which='both',
                        bottom=False, top=False, labelbottom=False,
                        right=False, left=False, labelleft=False)
    
        experiment_dir = os.path.join(os.getcwd(), 'results', 'experiment%02d' % experiment)
        plot_fn = os.path.join(experiment_dir, 'heat_map_%s.pdf' % agent_name)
        plt.savefig(plot_fn, bbox_inches='tight')
        
        plt.show()


if __name__ == '__main__':
    experiment = 36  # 36, 38
    agent_names = ['ase-agent', 'undirected-ase-agent', 'mbie-agent', 'safe-rmax-agent', 'rmax-agent', 'safe-egreedy-agent', 'egreedy-agent']
    plot_num_sub_optimal(agent_names, experiment)
    
    experiment = 38
    agent_names = ['ase-agent', 'mbie-agent', 'safe-rmax-agent']
    plot_heat_map(agent_names, experiment)

    experiment = 36
    experiment_dir = os.path.join(os.getcwd(), 'results', 'experiment%02d' % experiment, agent_names[0], 'trial%02d' % 1)
    param_dict_fn = os.path.join(experiment_dir, 'param_dict.pkl')
    param_dict = pickle.load(open(param_dict_fn, 'rb'))
    env = unsafe_grid_world.make_grid_world_from_file(param_dict['level'], slip_prob=0.6, gui=True)
    map_image = env.map_image()

    map_image_fn = os.path.join(os.getcwd(), 'results', 'experiment%02d' % experiment, 'map_image.png')
    cv2.imwrite(map_image_fn, cv2.cvtColor(map_image, cv2.COLOR_RGB2BGR))

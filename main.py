import os
import numpy as np
import pickle
from multiprocessing import Pool
import argparse
import yaml

from environments import discrete_platformer, unsafe_grid_world
from agents.ase_agent import ASEAgent
from agents.safe_rmax_agent import SafeRMaxAgent
from agents.safe_epsilon_greedy import SafeEGreedyAgent
from agents.mbie_agent import MBIEAgent


def setup_unsafe_grid_world_env(file_name, gui=False):
    env = unsafe_grid_world.make_grid_world_from_file(file_name, slip_prob=0.6, gui=gui)
    
    safe_states = env.get_safe_states()
    safe_actions_for_state = env.safe_actions
    similarity_function = lambda state_list, action_list: env.location_invariance(state_list, action_list, distance=5)
    analagous_state_function = lambda s, a, sp, s2, a2: env.location_invariance_equivalency(s, a, sp, s2, a2)
    transition_support_function = unsafe_grid_world.transition_support_function
    
    return env, safe_states, safe_actions_for_state, \
           similarity_function, analagous_state_function, \
           transition_support_function


def setup_discrete_platformer_env(file_name, gui=False):
    env, safe_locs = discrete_platformer.env_from_file(file_name, gui=gui)
    safe_states, safe_actions_for_state = safe_state_actions_for_discrete_platformer(env, safe_locs)
    initial_safe_action_function = lambda state: [] if state not in safe_actions_for_state else safe_actions_for_state[state]
    similarity_function = lambda state_list, action_list: env.horizontal_invariance(state_list, action_list)
    analagous_state_function = lambda s, a, sp, s2, a2: env.horizontal_invariance_equivalency(s, a, sp, s2, a2)
    transition_support_function = discrete_platformer.transition_support_function
    
    return env, safe_states, initial_safe_action_function, \
           similarity_function, analagous_state_function, \
           transition_support_function


def safe_state_actions_for_discrete_platformer(env, safe_locs):
    safe_states = set()
    actions_for_state = dict()
    all_actions = env.get_actions()
    for loc in safe_locs:
        x = loc[0]
        for x_dot in range(-discrete_platformer.PLAYER_MAX_SPEED, discrete_platformer.PLAYER_MAX_SPEED + 1):
            y = env.init_state.y
            state0 = discrete_platformer.PlatformerState(x, y, x_dot, 0, env.on_platform(x, y))
            safe_states.add(state0)
            actions_for_state[state0] = all_actions
            for action in all_actions:
                next_states, p = env.transition_function(state0, action)
                for next_state in next_states:
                    state = next_state
                    while True:
                        safe_states.add(state)
                        if env.on_platform(state.x, state.y) == 0:
                            actions = all_actions
                        elif state.x > env.init_state.x:
                            actions = [discrete_platformer.LEFT]
                        else:
                            actions = [discrete_platformer.RIGHT]
                        
                        if state not in actions_for_state:
                            actions_for_state[state] = actions
                        next_states2, _ = env.transition_function(state, actions[0])
                        assert len(next_states2) == 1
                        state = next_states2[0]
                        
                        if env.on_platform(state.x, state.y) > 0 and (state.x, state.y) in safe_locs:
                            break
    
    return safe_states, actions_for_state


def setup_ase_agent(name, env, initial_safe_set, initial_safe_action_function,
                    similarity_function, analagous_state_function,
                    transition_support_function, param_dict,
                    directed=True):
    return ASEAgent(actions=env.get_actions(), states=env.get_all_states(), reward_func=env.reward_function,
                    initial_safe_states=initial_safe_set,
                    initial_safe_actions=initial_safe_action_function,
                    similarity_function=similarity_function,
                    analagous_state_function=analagous_state_function,
                    transition_support_function=transition_support_function,
                    gamma=param_dict['gamma'], vi_horizon=param_dict['vi_horizon'],
                    tau=param_dict['tau'], update_frequency=param_dict['update_frequency'],
                    beta_T=param_dict['beta_T'],
                    use_sparse_matrices=param_dict['use_sparse_matrices'],
                    directed=directed,
                    name=name)


def setup_rmax_agent(name, env, initial_safe_set, initial_safe_action_function,
                     similarity_function, analagous_state_function,
                     transition_support_function, param_dict,
                     safe=True):
    return SafeRMaxAgent(actions=env.get_actions(), states=env.get_all_states(), reward_func=env.reward_function,
                         initial_safe_states=initial_safe_set,
                         initial_safe_actions=initial_safe_action_function,
                         similarity_function=similarity_function,
                         analagous_state_function=analagous_state_function,
                         transition_support_function=transition_support_function,
                         gamma=param_dict['gamma'], vi_horizon=param_dict['vi_horizon'],
                         tau=param_dict['tau'], update_frequency=param_dict['update_frequency'],
                         beta_T=param_dict['beta_T'],
                         use_sparse_matrices=param_dict['use_sparse_matrices'],
                         safe=safe,
                         name=name)


def setup_egreedy_agent(name, env, initial_safe_set, initial_safe_action_function,
                        similarity_function, analagous_state_function,
                        transition_support_function, param_dict,
                        safe=True):
    return SafeEGreedyAgent(actions=env.get_actions(), states=env.get_all_states(), reward_func=env.reward_function,
                            initial_safe_states=initial_safe_set,
                            initial_safe_actions=initial_safe_action_function,
                            similarity_function=similarity_function,
                            analagous_state_function=analagous_state_function,
                            transition_support_function=transition_support_function,
                            gamma=param_dict['gamma'], vi_horizon=param_dict['vi_horizon'],
                            epsilon=param_dict['epsilon'], annealing_time=param_dict['annealing_time'],
                            tau=param_dict['tau'], update_frequency=param_dict['update_frequency'],
                            beta_T=param_dict['beta_T'],
                            use_sparse_matrices=param_dict['use_sparse_matrices'],
                            safe=safe,
                            name=name)


def setup_mbie_agent(name, env, initial_safe_set, initial_safe_action_function,
                     similarity_function, analagous_state_function,
                     transition_support_function, param_dict):
    return MBIEAgent(actions=env.get_actions(), states=env.get_all_states(), reward_func=env.reward_function,
                     initial_safe_states=initial_safe_set,
                     initial_safe_actions=initial_safe_action_function,
                     similarity_function=similarity_function,
                     analagous_state_function=analagous_state_function,
                     transition_support_function=transition_support_function,
                     gamma=param_dict['gamma'], vi_horizon=param_dict['vi_horizon'],
                     tau=param_dict['tau'], update_frequency=param_dict['update_frequency'],
                     beta_T=param_dict['beta_T'],
                     use_sparse_matrices=param_dict['use_sparse_matrices'],
                     name=name)


def setup_agent(agent_name,
                env, initial_safe_set, initial_safe_action_function,
                similarity_function, analagous_state_function,
                transition_support_function, param_dict):
    
    if agent_name == 'ase-agent':
        return setup_ase_agent(agent_name, env, initial_safe_set, initial_safe_action_function,
                               similarity_function, analagous_state_function,
                               transition_support_function, param_dict, directed=True)
    if agent_name == 'undirected-ase-agent':
        return setup_ase_agent(agent_name, env, initial_safe_set, initial_safe_action_function,
                               similarity_function, analagous_state_function,
                               transition_support_function, param_dict, directed=False)
    if agent_name == 'mbie-agent':
        return setup_mbie_agent(agent_name, env, initial_safe_set, initial_safe_action_function,
                                similarity_function, analagous_state_function,
                                transition_support_function, param_dict)
    if agent_name == 'safe-rmax-agent':
        return setup_rmax_agent(agent_name, env, initial_safe_set, initial_safe_action_function,
                                similarity_function, analagous_state_function,
                                transition_support_function, param_dict,
                                safe=True)
    if agent_name == 'rmax-agent':
        return setup_rmax_agent(agent_name, env, initial_safe_set, initial_safe_action_function,
                                similarity_function, analagous_state_function,
                                transition_support_function, param_dict,
                                safe=False)
    if agent_name == 'safe-egreedy-agent':
        return setup_egreedy_agent(agent_name, env, initial_safe_set, initial_safe_action_function,
                                   similarity_function, analagous_state_function,
                                   transition_support_function, param_dict,
                                   safe=True)
    if agent_name == 'egreedy-agent':
        return setup_egreedy_agent(agent_name, env, initial_safe_set, initial_safe_action_function,
                                   similarity_function, analagous_state_function,
                                   transition_support_function, param_dict,
                                   safe=False)
    
    ValueError('No agent for agent id: %s' % agent_name)


def train(agent_name, environment_name, experiment=1, trial=1, max_steps=10000, gui=True, random_seed=123456):
    np.random.seed(random_seed + trial)
    
    if environment_name == 'discrete_platformer':
        temp = setup_discrete_platformer_env(param_dict['level'], gui=gui)
        env = temp[0]
    elif environment_name == 'unsafe_grid_world':
        temp = setup_unsafe_grid_world_env(param_dict['level'], gui=gui)
        env = temp[0]
    else:
        ValueError('Unrecognized Environment: %s' % environment_name)
        return
    agent = setup_agent(agent_name, *temp, param_dict)
    
    score = 0
    cur_state = env.get_current_state()
    reward = 0
    state_history = [cur_state]
    action_history = []
    reward_history = []
    z_safe_time = None
    if hasattr(agent, 'z_safe'):
        z_safe_time = np.where(agent.z_safe, 0, np.infty)
    
    for i in range(max_steps):
        if i % 1000 == 0:
            print('Step: %d' % i)
    
        # Move agent.
        action = agent.act(cur_state, reward)
        reward, cur_state = env.execute_agent_action(action)
        
        state_history.append(cur_state)
        action_history.append(action)
        reward_history.append(reward)
        if z_safe_time is not None:
            z_safe_time = np.minimum(z_safe_time, np.where(agent.z_safe, i, np.infty))
        
        score += int(reward)
        
        if cur_state.is_terminal():
            action = agent.act(cur_state, reward)
            reward, cur_state = env.execute_agent_action(action)
            score += int(reward)
            
            env.reset()
            cur_state = env.get_current_state()

            state_history.append(cur_state)
            action_history.append(action)
            reward_history.append(reward)
    
    experiment_dir = os.path.join(os.getcwd(), 'results',
                                  'experiment%02d' % experiment,
                                  agent.name,
                                  'trial%02d' % trial)
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    
    param_dict_fn = os.path.join(experiment_dir, 'param_dict.pkl')
    state_history_fn = os.path.join(experiment_dir, 'state_history.pkl')
    action_history_fn = os.path.join(experiment_dir, 'action_history.pkl')
    reward_history_fn = os.path.join(experiment_dir, 'reward_history.pkl')
    
    pickle.dump(param_dict, open(param_dict_fn, 'wb'))
    pickle.dump(state_history, open(state_history_fn, 'wb'))
    pickle.dump(action_history, open(action_history_fn, 'wb'))
    pickle.dump(reward_history, open(reward_history_fn, 'wb'))
    
    if z_safe_time is not None:
        z_safe_time_fn = os.path.join(experiment_dir, 'z_safe_time.pkl')
        pickle.dump(z_safe_time, open(z_safe_time_fn, 'wb'))


def run_experiment(param_dict):
    experiment_dir = os.path.join(os.getcwd(), 'results', 'experiment%02d' % param_dict['experiment'])
    while os.path.exists(experiment_dir):
        param_dict['experiment'] += 1
        experiment_dir = os.path.join(os.getcwd(), 'results', 'experiment%02d' % param_dict['experiment'])
    os.makedirs(experiment_dir)
    
    environment_name = param_dict['environment']
    
    num_threads = param_dict['num_threads']
    if num_threads > 1:
        pool = Pool(processes=num_threads)

    if param_dict['agents'] == 'all':
        agent_names = ['ase-agent', 'undirected-ase-agent', 'mbie-agent', 'safe-rmax-agent', 'rmax-agent', 'safe-egreedy-agent', 'egreedy-agent']
    else:
        agent_names = param_dict['agents']
    
    for agent_name in agent_names:
        print('Training agent: %s...' % agent_name)
        for t in range(1, param_dict['num_trials'] + 1):
            print('Starting trial: %02d...' % t)
            if num_threads > 1:
                pool.apply_async(train, (agent_name, environment_name, param_dict['experiment'], t, param_dict['max_steps'], False))
            else:
                train(agent_name, environment_name, experiment=param_dict['experiment'], trial=t, max_steps=param_dict['max_steps'], gui=True)

    if num_threads > 1:
        pool.close()
        pool.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run experiment')
    parser.add_argument('--config', type=str, default='experiment_configs/unsafe_grid_world.yaml',
                        help='path to a yaml config file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        param_dict = yaml.load(f)
    
    run_experiment(param_dict)

# -*- coding: utf-8 -*-
import itertools
import logging
import numpy as np
import tensorflow as tf
import time
import os
import pandas as pd
import subprocess


def check_dir(cur_dir):
    if not os.path.exists(cur_dir):
        return False
    return True


def copy_file(src_dir, tar_dir):
    cmd = 'cp -r %s %s/' % (src_dir, dest_dir)
    subprocess.check_call(cmd, shell=True)


def find_file(cur_dir, suffix='.ini'):
    for file in os.listdir(cur_dir):
        if file.endswith(suffix):
            return cur_dir + '/' + file
    logging.error('Cannot find %s file' % suffix)
    return None


def init_dir(base_dir, pathes=['log', 'data', 'model']):
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    dirs = {}
    for path in pathes:
        cur_dir = base_dir + '/%s/' % path
        if not os.path.exists(cur_dir):
            os.mkdir(cur_dir)
        dirs[path] = cur_dir
    return dirs


def init_log(log_dir):
    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                        level=logging.INFO,
                        handlers=[
                            logging.FileHandler('%s/%d.log' % (log_dir, time.time())),
                            logging.StreamHandler()
                        ])


def init_test_flag(test_mode):
    if test_mode == 'no_test':
        return False, False
    if test_mode == 'in_train_test':
        return True, False
    if test_mode == 'after_train_test':
        return False, True
    if test_mode == 'all_test':
        return True, True
    return False, False


def plot_train(data_dirs, labels):
    pass

def plot_evaluation(data_dirs, labels):
    pass


class Counter:
    def __init__(self, total_step, test_step, log_step):
        self.counter = itertools.count(1)
        self.cur_step = 0
        self.cur_test_step = 0
        self.total_step = total_step
        self.test_step = test_step
        self.log_step = log_step
        self.stop = False

    def next(self):
        self.cur_step = next(self.counter)
        return self.cur_step

    def should_test(self):
        test = False
        if (self.cur_step - self.cur_test_step) >= self.test_step:
            test = True
            self.cur_test_step = self.cur_step
        return test

    def should_log(self):
        return (self.cur_step % self.log_step == 0)

    def should_stop(self):
        if self.cur_step >= self.total_step:
            return True
        return self.stop


class Trainer():
    def __init__(self, env, model, global_counter, summary_writer, run_test, output_path=None):
        self.cur_step = 0
        self.global_counter = global_counter
        self.env = env
        self.agent = self.env.agent
        self.model = model
        self.sess = self.model.sess
        self.n_step = self.model.n_step
        self.summary_writer = summary_writer
        self.run_test = run_test
        self.is_ippo = env.agent == 'ippo'  
        self.is_sppo = env.agent == 'sppo'  # Added SPPO flag
        assert self.env.T % self.n_step == 0
        self.data = []
        self.output_path = output_path
        if run_test:
            self.test_num = self.env.test_num
            logging.info('Testing: total test num: %d' % self.test_num)
        self._init_summary()

    def _init_summary(self):
        self.train_reward = tf.placeholder(tf.float32, [])
        self.train_summary = tf.summary.scalar('train_reward', self.train_reward)
        self.test_reward = tf.placeholder(tf.float32, [])
        self.test_summary = tf.summary.scalar('test_reward', self.test_reward)

    def _add_summary(self, reward, global_step, is_train=True):
        if is_train:
            summ = self.sess.run(self.train_summary, {self.train_reward: reward})
        else:
            summ = self.sess.run(self.test_summary, {self.test_reward: reward})
        self.summary_writer.add_summary(summ, global_step=global_step)

    def _get_road_occupancy(self, from_node, to_node):
        """Get road occupancy from SUMO
        
        Parameters:
        from_node: Origin node ID (e.g., "nt1")
        to_node: Destination node ID (e.g., "nt2")
        
        Returns:
        occupancy: Road occupancy rate (value between 0-1)
        """
        # Construct road ID
        road_id = "{}_{}".format(from_node, to_node)
        
        # Get all lanes on the road
        try:
            lanes = self.env.sim.edge.getLaneNumber(road_id)
            total_occupancy = 0
            
            # Calculate occupancy rate for each lane
            for i in range(lanes):
                lane_id = "{}_{:d}".format(road_id, i)
                
                # Get lane length
                lane_length = self.env.sim.lane.getLength(lane_id)
                
                # Get all vehicles on the lane
                vehicles = self.env.sim.lane.getLastStepVehicleIDs(lane_id)
                
                # Calculate total length occupied by vehicles (vehicle length + safety distance)
                occupied_length = 0
                for veh_id in vehicles:
                    veh_length = self.env.sim.vehicle.getLength(veh_id)
                    # Add safety distance (approx. 2.5 meters)
                    occupied_length += veh_length + 2.5
                    
                # Calculate lane occupancy rate
                lane_occupancy = min(1.0, occupied_length / lane_length)
                total_occupancy += lane_occupancy
            
            # Return average occupancy rate
            return total_occupancy / max(1, lanes)
        except:
            # If error occurs (e.g., road doesn't exist), return 0
            return 0

    def _detect_spillback_from_sumo(self, node_name, neighbor_name):
        """Directly detect spillback between two nodes from SUMO
        
        Parameters:
        node_name: Current node name
        neighbor_name: Neighbor node name
        
        Returns:
        has_spillback: Boolean value indicating whether spillback exists
        """
        # Get road occupancy rate between the two nodes
        occupancy = self._get_road_occupancy(node_name, neighbor_name)
        
        # Determine if spillback exists
        # Consider spillback when occupancy exceeds 90%
        return occupancy > 0.9
    
    def _get_neighbors_info_from_sumo(self, obs):
        """Directly get neighbor node information from SUMO for spillback detection
        
        Parameters:
        obs: Observation list for all agents
        
        Returns:
        neighbors_info: List of neighbor information dictionaries for each agent
        """
        neighbors_info = []
        
        for i, ob in enumerate(obs):
            agent_neighbors = {}
            
            # Get node name (assume node ID is nt1, nt2, etc.)
            node_name = "nt{:d}".format(i+1)
            
            # Get neighbors of current node
            if hasattr(self.model.policy_ls[i], 'neighbor_indices'):
                neighbor_indices = self.model.policy_ls[i].neighbor_indices
                
                for j, neighbor_idx in enumerate(neighbor_indices):
                    if neighbor_idx < len(obs):
                        # Get neighbor node name
                        neighbor_name = "nt{:d}".format(neighbor_idx+1)
                        
                        # Detect if spillback exists
                        has_spillback = self._detect_spillback_from_sumo(node_name, neighbor_name)
                        
                        # Only retrieve neighbor state when spillback exists
                        if has_spillback:
                            agent_neighbors[neighbor_idx] = {
                                'has_spillback': True,
                                'state': obs[neighbor_idx]
                            }
                        else:
                            agent_neighbors[neighbor_idx] = {
                                'has_spillback': False,
                                'state': np.zeros_like(obs[neighbor_idx])
                            }
            
            neighbors_info.append(agent_neighbors)
        
        return neighbors_info

    def explore(self, prev_ob, prev_done):
        ob = prev_ob
        done = prev_done
        rewards = []
        for _ in range(self.n_step):
            if self.agent.endswith('a2c') or self.is_ippo or self.is_sppo:  # Include SPPO in condition
                if self.is_ippo:
                    # IPPO requires additional log probabilities
                    policy, value, log_probs = self.model.forward(ob, done, 'pvlogp')
                elif self.is_sppo:
                    # SPPO requires neighbor information
                    neighbors_info = self._get_neighbors_info_from_sumo(ob)
                    policy, value, log_probs, spillbacks = self.model.forward(ob, done, neighbors_info, 'pvlogp')
                else:
                    policy, value = self.model.forward(ob, done)
                
                # Need to update fingerprint before calling step
                if self.agent == 'ma2c':
                    self.env.update_fingerprint(policy)
                
                if self.agent == 'a2c':
                    action = np.random.choice(np.arange(len(policy)), p=policy)
                else:
                    action = []
                    for pi in policy:
                        action.append(np.random.choice(np.arange(len(pi)), p=pi))
            else:
                action, policy = self.model.forward(ob, mode='explore')
            
            next_ob, reward, done, global_reward = self.env.step(action)
            rewards.append(global_reward)
            global_step = self.global_counter.next()
            self.cur_step += 1
            
            # Call appropriate add_transition method based on model type
            if self.agent.endswith('a2c'):
                if not self.is_ippo and not self.is_sppo:
                    self.model.add_transition(ob, action, reward, value, done)
                elif self.is_ippo:
                    # IPPO requires additional log probability parameter
                    log_prob = []
                    for a, logp in zip(action, log_probs):
                        log_prob.append(logp[a])
                    self.model.add_transition(ob, action, reward, value, log_prob, done)
                elif self.is_sppo:
                    # SPPO requires additional log probability and spillback information
                    log_prob = []
                    for a, logp in zip(action, log_probs):
                        log_prob.append(logp[a])
                    self.model.add_transition(ob, action, reward, value, log_prob, done, spillbacks)
            else:
                # IQL-type models use different parameter list
                self.model.add_transition(ob, action, reward, next_ob, done)
            
            # Logging
            if self.global_counter.should_log():
                log_msg = '''Training: global step %d, episode step %d,
                             ob: %s, a: %s, pi: %s, r: %.2f, train r: %.2f, done: %r''' % \
                         (global_step, self.cur_step,
                          str(ob), str(action), str(policy), global_reward, np.mean(reward), done)
                logging.info(log_msg)
            
            if done:
                break
            ob = next_ob
        
        # Calculate return R based on model type
        if self.agent.endswith('a2c') or self.is_ippo or self.is_sppo:  # Include SPPO in condition
            if done:
                R = 0 if self.agent == 'a2c' else [0] * self.model.n_agent
            else:
                if self.is_sppo:
                    # SPPO requires neighbor information
                    neighbors_info = self._get_neighbors_info_from_sumo(ob)
                    R, _ = self.model.forward(ob, False, neighbors_info, 'v')
                else:
                    R = self.model.forward(ob, False, 'v')
        else:
            R = 0
        return ob, done, R, rewards
        
    def run_thread(self, coord):
        '''Multi-threading is disabled'''
        ob = self.env.reset()
        done = False
        cum_reward = 0
        while not coord.should_stop():
            ob, done, R, cum_reward = self.explore(ob, done, cum_reward)
            global_step = self.global_counter.cur_step
            if self.agent.endswith('a2c') or self.is_ippo or self.is_sppo:  # Include SPPO
                self.model.backward(R, self.summary_writer, global_step)
            else:
                self.model.backward(self.summary_writer, global_step)
            self.summary_writer.flush()
            if (self.global_counter.should_stop()) and (not coord.should_stop()):
                self.env.terminate()
                coord.request_stop()
                logging.info('Training: stop condition reached!')
                return

    def perform(self, test_ind, demo=False, policy_type='default'):
        """Used for testing phase."""
        ob = self.env.reset(gui=demo, test_ind=test_ind)
        # Note this done is pre-decision, used to reset LSTM states
        done = True
        self.model.reset()
        rewards = []
        while True:
            if self.agent == 'greedy':
                action = self.model.forward(ob)
            elif self.agent.endswith('a2c') or self.is_ippo or self.is_sppo:  # Modified to include SPPO
                # On-policy learning based on policy
                if self.is_sppo:
                    # SPPO needs additional neighbor information
                    neighbors_info = self._get_neighbors_info_from_sumo(ob)
                    policy, spillbacks = self.model.forward(ob, done, neighbors_info, 'p')
                else:
                    policy = self.model.forward(ob, done, 'p')
                    
                if self.agent == 'ma2c':
                    self.env.update_fingerprint(policy)
                if self.agent == 'a2c':
                    if policy_type != 'deterministic':
                        action = np.random.choice(np.arange(len(policy)), p=policy)
                    else:
                        action = np.argmax(np.array(policy))
                else:
                    action = []
                    for pi in policy:
                        if policy_type != 'deterministic':
                            action.append(np.random.choice(np.arange(len(pi)), p=pi))
                        else:
                            action.append(np.argmax(np.array(pi)))
            else:
                # Off-policy learning based on value
                if policy_type != 'stochastic':
                    action, _ = self.model.forward(ob)
                else:
                    action, _ = self.model.forward(ob, stochastic=True)
            next_ob, reward, done, global_reward = self.env.step(action)
            rewards.append(global_reward)
            if done:
                break
            ob = next_ob
        mean_reward = np.mean(np.array(rewards))
        std_reward = np.std(np.array(rewards))
        return mean_reward, std_reward

    def run(self):
        """Main training loop."""
        while not self.global_counter.should_stop():
            # Testing
            if self.run_test and self.global_counter.should_test():
                rewards = []
                global_step = self.global_counter.cur_step
                self.env.train_mode = False
                for test_ind in range(self.test_num):
                    mean_reward, std_reward = self.perform(test_ind)
                    self.env.terminate()
                    rewards.append(mean_reward)
                    log = {'agent': self.agent,
                           'step': global_step,
                           'test_id': test_ind,
                           'avg_reward': mean_reward,
                           'std_reward': std_reward}
                    self.data.append(log)
                avg_reward = np.mean(np.array(rewards))
                self._add_summary(avg_reward, global_step, is_train=False)
                logging.info('Testing: global step %d, avg R: %.2f' %
                             (global_step, avg_reward))
            
            # Training
            self.env.train_mode = True
            ob = self.env.reset()
            # Note this done is pre-decision, used to reset LSTM states
            done = True
            self.model.reset()
            self.cur_step = 0
            rewards = []
            while True:
                ob, done, R, cur_rewards = self.explore(ob, done)
                rewards += cur_rewards
                global_step = self.global_counter.cur_step
                if self.agent.endswith('a2c') or self.is_ippo or self.is_sppo:  # Modified to include SPPO
                    self.model.backward(R, self.summary_writer, global_step)
                else:
                    self.model.backward(self.summary_writer, global_step)
                # Termination condition
                if done:
                    self.env.terminate()
                    break
            
            rewards = np.array(rewards)
            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards)
            log = {'agent': self.agent,
                   'step': global_step,
                   'test_id': -1,
                   'avg_reward': mean_reward,
                   'std_reward': std_reward}
            self.data.append(log)
            self._add_summary(mean_reward, global_step)
            self.summary_writer.flush()
        
        df = pd.DataFrame(self.data)
        df.to_csv(self.output_path + 'train_reward.csv')


class Tester(Trainer):
    def __init__(self, env, model, global_counter, summary_writer, output_path):
        super().__init__(env, model, global_counter, summary_writer)
        self.env.train_mode = False
        self.test_num = self.env.test_num
        self.output_path = output_path
        self.data = []
        logging.info('Testing: total test num: %d' % self.test_num)

    def _init_summary(self):
        self.reward = tf.placeholder(tf.float32, [])
        self.summary = tf.summary.scalar('test_reward', self.reward)

    def run_offline(self):
        # enable traffic measurments for offline test
        is_record = True
        record_stats = False
        self.env.cur_episode = 0
        self.env.init_data(is_record, record_stats, self.output_path)
        rewards = []
        for test_ind in range(self.test_num):
            rewards.append(self.perform(test_ind))
            self.env.terminate()
            time.sleep(2)
            self.env.collect_tripinfo()
        avg_reward = np.mean(np.array(rewards))
        logging.info('Offline testing: avg R: %.2f' % avg_reward)
        self.env.output_data()

    def run_online(self, coord):
        self.env.cur_episode = 0
        while not coord.should_stop():
            time.sleep(30)
            if self.global_counter.should_test():
                rewards = []
                global_step = self.global_counter.cur_step
                for test_ind in range(self.test_num):
                    cur_reward = self.perform(test_ind)
                    self.env.terminate()
                    rewards.append(cur_reward)
                    log = {'agent': self.agent,
                           'step': global_step,
                           'test_id': test_ind,
                           'reward': cur_reward}
                    self.data.append(log)
                avg_reward = np.mean(np.array(rewards))
                self._add_summary(avg_reward, global_step)
                logging.info('Testing: global step %d, avg R: %.2f' %
                             (global_step, avg_reward))
        df = pd.DataFrame(self.data)
        df.to_csv(self.output_path + 'train_reward.csv')


class Evaluator(Tester):
    def __init__(self, env, model, output_path, demo=False, policy_type='default'):
        self.env = env
        self.model = model
        self.agent = self.env.agent
        self.env.train_mode = False
        self.test_num = self.env.test_num
        self.output_path = output_path
        self.demo = demo
        self.policy_type = policy_type

    def run(self):
        is_record = True
        record_stats = False
        self.env.cur_episode = 0
        self.env.init_data(is_record, record_stats, self.output_path)
        time.sleep(1)
        for test_ind in range(self.test_num):
            reward, _ = self.perform(test_ind, demo=self.demo, policy_type=self.policy_type)
            self.env.terminate()
            logging.info('test %i, avg reward %.2f' % (test_ind, reward))
            time.sleep(2)
            self.env.collect_tripinfo()
        self.env.output_data()
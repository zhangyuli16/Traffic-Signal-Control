# -*- coding: utf-8 -*-

import os
from agents.utils import *
from agents.policies import *
import logging
import multiprocessing as mp
import numpy as np
import tensorflow as tf


class A2C:
    def __init__(self, n_s, n_a, total_step, model_config, seed=0, n_f=None):
        # load parameters
        self.name = 'a2c'
        self.n_agent = 1
        # init reward norm/clip
        self.reward_clip = model_config.getfloat('reward_clip')
        self.reward_norm = model_config.getfloat('reward_norm')
        self.n_s = n_s
        self.n_a = n_a
        self.n_step = model_config.getint('batch_size')
        # init tf
        tf.reset_default_graph()
        tf.set_random_seed(seed)
        config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(config=config)
        self.policy = self._init_policy(n_s, n_a, n_f, model_config)
        self.saver = tf.train.Saver(max_to_keep=5)
        if total_step:
            # training
            self.total_step = total_step
            self._init_scheduler(model_config)
            self._init_train(model_config)
        self.sess.run(tf.global_variables_initializer())

    def _init_policy(self, n_s, n_a, n_w, n_f, model_config, agent_name=None):
        n_fw = model_config.getint('num_fw')
        n_ft = model_config.getint('num_ft')
        n_lstm = model_config.getint('num_lstm')
        if self.name == 'ma2c':
            n_fp = model_config.getint('num_fp')
            policy = FPLstmACPolicy(n_s, n_a, n_w, n_f, self.n_step, n_fc_wave=n_fw,
                                    n_fc_wait=n_ft, n_fc_fp=n_fp, n_lstm=n_lstm, name=agent_name)
        else:
            policy = LstmACPolicy(n_s, n_a, n_w, self.n_step, n_fc_wave=n_fw,
                                  n_fc_wait=n_ft, n_lstm=n_lstm, name=agent_name)
        return policy

    def _init_scheduler(self, model_config):
        lr_init = model_config.getfloat('lr_init')
        lr_decay = model_config.get('lr_decay')
        beta_init = model_config.getfloat('entropy_coef_init')
        beta_decay = model_config.get('entropy_decay')
        if lr_decay == 'constant':
            self.lr_scheduler = Scheduler(lr_init, decay=lr_decay)
        else:
            lr_min = model_config.getfloat('LR_MIN')
            self.lr_scheduler = Scheduler(lr_init, lr_min, self.total_step, decay=lr_decay)
        if beta_decay == 'constant':
            self.beta_scheduler = Scheduler(beta_init, decay=beta_decay)
        else:
            beta_min = model_config.getfloat('ENTROPY_COEF_MIN')
            beta_ratio = model_config.getfloat('ENTROPY_RATIO')
            self.beta_scheduler = Scheduler(beta_init, beta_min, self.total_step * beta_ratio,
                                            decay=beta_decay)

    def _init_train(self, model_config):
        # init loss
        v_coef = model_config.getfloat('value_coef')
        max_grad_norm = model_config.getfloat('max_grad_norm')
        alpha = model_config.getfloat('rmsp_alpha')
        epsilon = model_config.getfloat('rmsp_epsilon')
        self.policy.prepare_loss(v_coef, max_grad_norm, alpha, epsilon)

        # init replay buffer
        gamma = model_config.getfloat('gamma')
        self.trans_buffer = OnPolicyBuffer(gamma)

    def save(self, model_dir, global_step):
        self.saver.save(self.sess, model_dir + 'checkpoint', global_step=global_step)

    def load(self, model_dir, checkpoint=None):
        save_file = None
        save_step = 0
        if os.path.exists(model_dir):
            if checkpoint is None:
                for file in os.listdir(model_dir):
                    if file.startswith('checkpoint'):
                        prefix = file.split('.')[0]
                        tokens = prefix.split('-')
                        if len(tokens) != 2:
                            continue
                        cur_step = int(tokens[1])
                        if cur_step > save_step:
                            save_file = prefix
                            save_step = cur_step
            else:
                save_file = 'checkpoint-' + str(int(checkpoint))
        if save_file is not None:
            self.saver.restore(self.sess, model_dir + save_file)
            logging.info('Checkpoint loaded: %s' % save_file)
            return True
        logging.error('Can not find old checkpoint for %s' % model_dir)
        return False

    def reset(self):
        self.policy._reset()

    def backward(self, R, summary_writer=None, global_step=None):
        cur_lr = self.lr_scheduler.get(self.n_step)
        cur_beta = self.beta_scheduler.get(self.n_step)
        obs, acts, dones, Rs, Advs = self.trans_buffer.sample_transition(R)
        self.policy.backward(self.sess, obs, acts, dones, Rs, Advs, cur_lr, cur_beta,
                             summary_writer=summary_writer, global_step=global_step)

    def forward(self, ob, done, out_type='pv'):
        return self.policy.forward(self.sess, ob, done, out_type)

    def add_transition(self, ob, action, reward, value, done):
        # Hard code the reward norm for negative reward only
        if (self.reward_norm):
            reward /= self.reward_norm
        if self.reward_clip:
            reward = np.clip(reward, -self.reward_clip, self.reward_clip)
        self.trans_buffer.add_transition(ob, action, reward, value, done)


class IA2C(A2C):
    def __init__(self, n_s_ls, n_a_ls, n_w_ls, total_step,
                 model_config, seed=0):
        self.name = 'ia2c'
        self.agents = []
        self.n_agent = len(n_s_ls)
        self.reward_clip = model_config.getfloat('reward_clip')
        self.reward_norm = model_config.getfloat('reward_norm')
        self.n_s_ls = n_s_ls
        self.n_a_ls = n_a_ls
        self.n_w_ls = n_w_ls
        self.n_step = model_config.getint('batch_size')
        # init tf
        tf.reset_default_graph()
        tf.set_random_seed(seed)
        config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(config=config)
        self.policy_ls = []
        for i, (n_s, n_w, n_a) in enumerate(zip(self.n_s_ls, self.n_w_ls, self.n_a_ls)):
            # agent_name is needed to differentiate multi-agents
            self.policy_ls.append(self._init_policy(n_s - n_w, n_a, n_w, 0, model_config,
                                  agent_name='{:d}a'.format(i)))
        self.saver = tf.train.Saver(max_to_keep=5)
        if total_step:
            # training
            self.total_step = total_step
            self._init_scheduler(model_config)
            self._init_train(model_config)
        self.sess.run(tf.global_variables_initializer())

    def _init_train(self, model_config):
        # init loss
        v_coef = model_config.getfloat('value_coef')
        max_grad_norm = model_config.getfloat('max_grad_norm')
        alpha = model_config.getfloat('rmsp_alpha')
        epsilon = model_config.getfloat('rmsp_epsilon')
        gamma = model_config.getfloat('gamma')
        self.trans_buffer_ls = []
        for i in range(self.n_agent):
            self.policy_ls[i].prepare_loss(v_coef, max_grad_norm, alpha, epsilon)
            self.trans_buffer_ls.append(OnPolicyBuffer(gamma))

    def backward(self, R_ls, summary_writer=None, global_step=None):
        cur_lr = self.lr_scheduler.get(self.n_step)
        cur_beta = self.beta_scheduler.get(self.n_step)
        for i in range(self.n_agent):
            obs, acts, dones, Rs, Advs = self.trans_buffer_ls[i].sample_transition(R_ls[i])
            if i == 0:
                self.policy_ls[i].backward(self.sess, obs, acts, dones, Rs, Advs, cur_lr, cur_beta,
                                           summary_writer=summary_writer, global_step=global_step)
            else:
                self.policy_ls[i].backward(self.sess, obs, acts, dones, Rs, Advs, cur_lr, cur_beta)

    def forward(self, obs, done, out_type='pv'):
        if len(out_type) == 1:
            out = []
        elif len(out_type) == 2:
            out1, out2 = [], []
        for i in range(self.n_agent):
            cur_out = self.policy_ls[i].forward(self.sess, obs[i], done, out_type)
            if len(out_type) == 1:
                out.append(cur_out)
            else:
                out1.append(cur_out[0])
                out2.append(cur_out[1])
        if len(out_type) == 1:
            return out
        else:
            return out1, out2

    def backward_mp(self, R_ls, summary_writer=None, global_step=None):
        cur_lr = self.lr_scheduler.get(self.n_step)
        cur_beta = self.beta_scheduler.get(self.n_step)

        def worker(i):
            obs, acts, dones, Rs, Advs = self.trans_buffer_ls[i].sample_transition(R_ls[i])
            self.policy_ls[i].backward(self.sess, obs, acts, dones, Rs, Advs, cur_lr, cur_beta,
                                       summary_writer=summary_writer, global_step=global_step)
        mps = []
        for i in range(self.n_agent):
            p = mp.Process(target=worker, args=(i))
            p.start()
            mps.append(p)
        for p in mps:
            p.join()

    def reset(self):
        for policy in self.policy_ls:
            policy._reset()

    def add_transition(self, obs, actions, rewards, values, done):
        if (self.reward_norm):
            rewards = rewards / self.reward_norm
        if self.reward_clip:
            rewards = np.clip(rewards, -self.reward_clip, self.reward_clip)
        for i in range(self.n_agent):
            self.trans_buffer_ls[i].add_transition(obs[i], actions[i],
                                                   rewards[i], values[i], done)


class MA2C(IA2C):
    def __init__(self, n_s_ls, n_a_ls, n_w_ls, n_f_ls, total_step,
                 model_config, seed=0):
        self.name = 'ma2c'
        self.agents = []
        self.n_agent = len(n_s_ls)
        self.reward_clip = model_config.getfloat('reward_clip')
        self.reward_norm = model_config.getfloat('reward_norm')
        self.n_s_ls = n_s_ls
        self.n_a_ls = n_a_ls
        self.n_f_ls = n_f_ls
        self.n_w_ls = n_w_ls
        self.n_step = model_config.getint('batch_size')
        # init tf
        tf.reset_default_graph()
        tf.set_random_seed(seed)
        config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(config=config)
        self.policy_ls = []
        for i, (n_s, n_a, n_w, n_f) in enumerate(zip(self.n_s_ls, self.n_a_ls, self.n_w_ls, self.n_f_ls)):
            # agent_name is needed to differentiate multi-agents
            self.policy_ls.append(self._init_policy(n_s - n_f - n_w, n_a, n_w, n_f, model_config,
                                                    agent_name='{:d}a'.format(i)))
        self.saver = tf.train.Saver(max_to_keep=5)
        if total_step:
            # training
            self.total_step = total_step
            self._init_scheduler(model_config)
            self._init_train(model_config)
        self.sess.run(tf.global_variables_initializer())

class IPPO(IA2C):
    """Independent PPOÃÃ£Â·Â¨ÃÂµÃÃÂ¡Â£
    ÃÂ¿Â¸Ã¶ÃÃÃÃÃÃ¥ÃÂ¹ÃÃÂ¶ÃÃÂ¢ÂµÃPPOÃÃ£Â·Â¨ÃÂ§ÃÂ°Â¡Â£
    """
    def __init__(self, n_s_ls, n_a_ls, n_w_ls, total_step,
                 model_config, seed=0):
        self.name = 'ippo'
        self.agents = []
        self.n_agent = len(n_s_ls)
        self.reward_clip = model_config.getfloat('reward_clip')
        self.reward_norm = model_config.getfloat('reward_norm')
        self.n_s_ls = n_s_ls
        self.n_a_ls = n_a_ls
        self.n_w_ls = n_w_ls
        self.n_step = model_config.getint('batch_size')
        
        # PPOÃÃÃÃÂ²ÃÃÃ½
        self.clip_ratio = model_config.getfloat('clip_ratio', 0.2)
        self.vf_coef = model_config.getfloat('value_coef')
        self.ent_coef = model_config.getfloat('entropy_coef_init')
        self.n_epochs = model_config.getint('n_epochs', 4)
        
        # Â³ÃµÃÂ¼Â»Â¯TensorFlow
        tf.reset_default_graph()
        tf.set_random_seed(seed)
        config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(config=config)
        self.policy_ls = []
        for i, (n_s, n_w, n_a) in enumerate(zip(self.n_s_ls, self.n_w_ls, self.n_a_ls)):
            # agent_nameÃÃÃÃÃÃ¸Â·ÃÂ¶Ã Â¸Ã¶ÃÃÃÃÃÃ¥
            self.policy_ls.append(self._init_policy(n_s - n_w, n_a, n_w, 0, model_config,
                                  agent_name='{:d}a'.format(i)))
        self.saver = tf.train.Saver(max_to_keep=5)
        if total_step:
            # ÃÂµÃÂ·
            self.total_step = total_step
            self._init_scheduler(model_config)
            self._init_train(model_config)
        self.sess.run(tf.global_variables_initializer())

    def _init_policy(self, n_s, n_a, n_w, n_f, model_config, agent_name=None):
        n_fw = model_config.getint('num_fw')
        n_ft = model_config.getint('num_ft')
        n_lstm = model_config.getint('num_lstm')
        # ÃÂ¹ÃÃÃÃ«IA2CÃÃ ÃÂ¬ÂµÃÃÃ¸ÃÃ§Â½Ã¡Â¹Â¹Â£Â¬ÂµÂ«ÃÂ¹ÃÃPPOPolicy
        policy = PPOPolicy(n_s, n_a, n_w, self.n_step, n_fc_wave=n_fw,
                           n_fc_wait=n_ft, n_lstm=n_lstm, name=agent_name)
        return policy

    def _init_train(self, model_config):
        # Â³ÃµÃÂ¼Â»Â¯ÃÃ°ÃÂ§ÂºÂ¯ÃÃ½
        v_coef = model_config.getfloat('value_coef')
        max_grad_norm = model_config.getfloat('max_grad_norm')
        alpha = model_config.getfloat('rmsp_alpha')
        epsilon = model_config.getfloat('rmsp_epsilon')
        gamma = model_config.getfloat('gamma')
        self.trans_buffer_ls = []
        for i in range(self.n_agent):
            self.policy_ls[i].prepare_loss(v_coef, max_grad_norm, alpha, epsilon, 
                                          self.clip_ratio, self.ent_coef)
            self.trans_buffer_ls.append(PPOBuffer(gamma, self.n_step))

    def add_transition(self, obs, actions, rewards, values, log_probs, done):
        if (self.reward_norm):
            rewards = rewards / self.reward_norm
        if self.reward_clip:
            rewards = np.clip(rewards, -self.reward_clip, self.reward_clip)
        for i in range(self.n_agent):
            self.trans_buffer_ls[i].add_transition(obs[i], actions[i],
                                                  rewards[i], values[i], 
                                                  log_probs[i], done)

    def backward(self, R_ls, summary_writer=None, global_step=None):
        cur_lr = self.lr_scheduler.get(self.n_step)
        
        # Â¶ÃÃÂ¿Â¸Ã¶ÃÃÃÃÃÃ¥Â½Ã¸ÃÃÂ¶Ã Â¸Ã¶epochsÂµÃÃÂµÃÂ·
        for i in range(self.n_agent):
            data = self.trans_buffer_ls[i].sample_transition(R_ls[i])
            # Â¶Ã ÃÃÃÂµÃÂ·
            for _ in range(self.n_epochs):
                if i == 0 and _ == 0:  # ÃÂ»ÃÃÂµÃÃÂ»Â¸Ã¶ÃÃÃÃÃÃ¥ÂµÃÂµÃÃÂ»Â¸Ã¶epochÂ¼ÃÃÂ¼ÃÂªÃÂª
                    self.policy_ls[i].backward(self.sess, *data, cur_lr,
                                              summary_writer=summary_writer, 
                                              global_step=global_step)
                else:
                    self.policy_ls[i].backward(self.sess, *data, cur_lr)

    def forward(self, obs, done, out_type='pv'):
        """ÃÂ°ÃÃ²Â´Â«Â²Â¥Â£Â¬Â¸Ã¹Â¾Ãout_typeÂ·ÂµÂ»ÃÂ²Â»ÃÂ¬ÃÃ ÃÃÂµÃÃÃ¤Â³Ã¶
        'p': Â·ÂµÂ»ÃÂ²ÃÃÃÂ£Â¨Â¶Â¯ÃÃ·Â¸ÃÃÃÂ·ÃÂ²Â¼Â£Â©
        'v': Â·ÂµÂ»ÃÃÂµÂºÂ¯ÃÃ½
        'pv': Â·ÂµÂ»ÃÂ²ÃÃÃÂºÃÃÂµÂºÂ¯ÃÃ½
        'pvlogp': Â·ÂµÂ»ÃÂ²ÃÃÃÂ¡Â¢ÃÂµÂºÂ¯ÃÃ½ÂºÃlogÂ¸ÃÃÃ
        """
        if out_type == 'pvlogp':
            # Â»Ã±ÃÂ¡Â²ÃÃÃÂ¡Â¢ÃÂµÂºÂ¯ÃÃ½ÂºÃÂ¶Â¯ÃÃ·ÂµÃlogÂ¸ÃÃÃ
            policies, values = [], []
            log_probs = []
            for i in range(self.n_agent):
                pi, v, logp = self.policy_ls[i].forward(self.sess, obs[i], done, 'pvlogp')
                policies.append(pi)
                values.append(v)
                log_probs.append(logp)
            return policies, values, log_probs
        else:
            # ÂµÃ·ÃÃÂ¸Â¸ÃÃ ÂµÃforwardÂ·Â½Â·Â¨Â´Â¦ÃÃ­ÃÃ¤ÃÃ»ÃÃ©Â¿Ã¶
            return super().forward(obs, done, out_type)


class IQL(A2C):
    def __init__(self, n_s_ls, n_a_ls, n_w_ls, total_step, model_config, seed=0, model_type='dqn'):
        self.name = 'iql'
        self.model_type = model_type
        self.agents = []
        self.n_agent = len(n_s_ls)
        self.reward_clip = model_config.getfloat('reward_clip')
        self.reward_norm = model_config.getfloat('reward_norm')
        self.n_s_ls = n_s_ls
        self.n_a_ls = n_a_ls
        self.n_w_ls = n_w_ls
        self.n_step = model_config.getint('batch_size')
        # init tf
        tf.reset_default_graph()
        tf.set_random_seed(seed)
        config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(config=config)
        self.policy_ls = []
        for i, (n_s, n_a, n_w) in enumerate(zip(self.n_s_ls, self.n_a_ls, self.n_w_ls)):
            # agent_name is needed to differentiate multi-agents
            self.policy_ls.append(self._init_policy(n_s, n_a, n_w, model_config,
                                                    agent_name='{:d}a'.format(i)))
        self.saver = tf.train.Saver(max_to_keep=5)
        if total_step:
            # training
            self.total_step = total_step
            self._init_scheduler(model_config)
            self._init_train(model_config)
        self.cur_step = 0
        self.sess.run(tf.global_variables_initializer())

    def _init_policy(self, n_s, n_a, n_w, model_config, agent_name=None):
        if self.model_type == 'dqn':
            n_h = model_config.getint('num_h')
            n_fc = model_config.getint('num_fc')
            policy = DeepQPolicy(n_s - n_w, n_a, n_w, self.n_step, n_fc0=n_fc, n_fc=n_h,
                                 name=agent_name)
        else:
            policy = LRQPolicy(n_s, n_a, self.n_step, name=agent_name)
        return policy

    def _init_scheduler(self, model_config):
        lr_init = model_config.getfloat('lr_init')
        lr_decay = model_config.get('lr_decay')
        eps_init = model_config.getfloat('epsilon_init')
        eps_decay = model_config.get('epsilon_decay')
        if lr_decay == 'constant':
            self.lr_scheduler = Scheduler(lr_init, decay=lr_decay)
        else:
            lr_min = model_config.getfloat('lr_min')
            self.lr_scheduler = Scheduler(lr_init, lr_min, self.total_step, decay=lr_decay)
        if eps_decay == 'constant':
            self.eps_scheduler = Scheduler(eps_init, decay=eps_decay)
        else:
            eps_min = model_config.getfloat('epsilon_min')
            eps_ratio = model_config.getfloat('epsilon_ratio')
            self.eps_scheduler = Scheduler(eps_init, eps_min, self.total_step * eps_ratio,
                                           decay=eps_decay)

    def _init_train(self, model_config):
        # init loss
        max_grad_norm = model_config.getfloat('max_grad_norm')
        gamma = model_config.getfloat('gamma')
        buffer_size = model_config.getfloat('buffer_size')
        self.trans_buffer_ls = []
        for i in range(self.n_agent):
            self.policy_ls[i].prepare_loss(max_grad_norm, gamma)
            self.trans_buffer_ls.append(ReplayBuffer(buffer_size, self.n_step))

    def backward(self, summary_writer=None, global_step=None):
        cur_lr = self.lr_scheduler.get(self.n_step)
        if self.trans_buffer_ls[0].size < self.trans_buffer_ls[0].batch_size:
            return
        for i in range(self.n_agent):
            for k in range(10):
                obs, acts, next_obs, rs, dones = self.trans_buffer_ls[i].sample_transition()
                if i == 0:
                    self.policy_ls[i].backward(self.sess, obs, acts, next_obs, dones, rs, cur_lr,
                                               summary_writer=summary_writer,
                                               global_step=global_step + k)
                else:
                    self.policy_ls[i].backward(self.sess, obs, acts, next_obs, dones, rs, cur_lr)

    def forward(self, obs, mode='act', stochastic=False):
        if mode == 'explore':
            eps = self.eps_scheduler.get(1)
        action = []
        qs_ls = []
        for i in range(self.n_agent):
            qs = self.policy_ls[i].forward(self.sess, obs[i])
            if (mode == 'explore') and (np.random.random() < eps):
                action.append(np.random.randint(self.n_a_ls[i]))
            else:
                if not stochastic:
                    action.append(np.argmax(qs))
                else:
                    qs = qs / np.sum(qs)
                    action.append(np.random.choice(np.arange(len(qs)), p=qs))
            qs_ls.append(qs)
        return action, qs_ls

    def reset(self):
        # do nothing
        return

    def add_transition(self, obs, actions, rewards, next_obs, done):
        """Ensure parameter list matches the call site
        
        Parameters:
        obs: Current state observations
        actions: Actions taken
        rewards: Rewards received
        next_obs: Next states
        done: Terminal flag
        """
        if (self.reward_norm):
            rewards = rewards / self.reward_norm
        if self.reward_clip:
            rewards = np.clip(rewards, -self.reward_clip, self.reward_clip)
        for i in range(self.n_agent):
            self.trans_buffer_ls[i].add_transition(obs[i], actions[i],
                                                  rewards[i], next_obs[i], done)
                                                   
                                                   
                                                   
class SPPO(IA2C):
    """Spillback-aware Proximal Policy Optimizationç®æ³å®ç°ã
    æ¯ä¸ªæºè½ä½ä½¿ç¨ç¬ç«çPPOç®æ³å­¦ä¹ ï¼å¹¶èèé»è¿è·¯å£é´çæº¢åºç°è±¡ã
    åªæå½éè·¯çæ­£å¡æ»¡è½¦æ¶æè·åé»å±ä¿¡æ¯ã
    """
    def __init__(self, n_s_ls, n_a_ls, n_w_ls, total_step,
                 model_config, seed=0):
        self.name = 'sppo'
        self.agents = []
        self.n_agent = len(n_s_ls)
        self.reward_clip = model_config.getfloat('reward_clip')
        self.reward_norm = model_config.getfloat('reward_norm')
        self.n_s_ls = n_s_ls
        self.n_a_ls = n_a_ls
        self.n_w_ls = n_w_ls
        self.n_step = model_config.getint('batch_size')
        
        # PPOç¹æåæ°
        self.clip_ratio = model_config.getfloat('clip_ratio')
        self.vf_coef = model_config.getfloat('value_coef')
        self.ent_coef = model_config.getfloat('entropy_coef_init')
        self.n_epochs = model_config.getint('n_epochs')
        
        # åå§åTensorFlow
        tf.reset_default_graph()
        tf.set_random_seed(seed)
        config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(config=config)
        self.policy_ls = []
        for i, (n_s, n_w, n_a) in enumerate(zip(self.n_s_ls, self.n_w_ls, self.n_a_ls)):
            # agent_nameç¨äºåºåå¤ä¸ªæºè½ä½
            self.policy_ls.append(self._init_policy(n_s - n_w, n_a, n_w, 0, model_config,
                                  agent_name='{:d}a'.format(i)))
        self.saver = tf.train.Saver(max_to_keep=5)
        if total_step:
            # è®­ç»
            self.total_step = total_step
            self._init_scheduler(model_config)
            self._init_train(model_config)
        self.sess.run(tf.global_variables_initializer())

    def _init_policy(self, n_s, n_a, n_w, n_f, model_config, agent_name=None):
        n_fw = model_config.getint('num_fw')
        n_ft = model_config.getint('num_ft')
        n_lstm = model_config.getint('num_lstm')
        
        # ä½¿ç¨SPPOPolicyï¼å®ç»§æ¿èªPPOPolicyä½ææº¢åºæç¥åè½
        policy = SPPOPolicy(n_s, n_a, n_w, self.n_step, n_fc_wave=n_fw,
                          n_fc_wait=n_ft, n_lstm=n_lstm, name=agent_name)
        return policy

    def _init_train(self, model_config):
        # åå§åæå¤±å½æ°
        v_coef = model_config.getfloat('value_coef')
        max_grad_norm = model_config.getfloat('max_grad_norm')
        alpha = model_config.getfloat('rmsp_alpha')
        epsilon = model_config.getfloat('rmsp_epsilon')
        gamma = model_config.getfloat('gamma')
        self.trans_buffer_ls = []
        for i in range(self.n_agent):
            self.policy_ls[i].prepare_loss(v_coef, max_grad_norm, alpha, epsilon, 
                                          self.clip_ratio, self.ent_coef)
            self.trans_buffer_ls.append(SPPOBuffer(gamma, self.n_step))

    def add_transition(self, obs, actions, rewards, values, log_probs, done, spillbacks=None):
        """Add a transition with spillback information
        
        Parameters:
        obs: Current state observations
        actions: Actions taken
        rewards: Rewards received
        values: Value estimates
        log_probs: Log probabilities of actions
        done: Terminal flag
        spillbacks: Spillback detection information (optional)
        """
        if (self.reward_norm):
            rewards = rewards / self.reward_norm
        if self.reward_clip:
            rewards = np.clip(rewards, -self.reward_clip, self.reward_clip)
        
        # If no spillbacks are provided, create default zero arrays
        if spillbacks is None:
            spillbacks = [np.zeros(len(self.policy_ls[i].neighbor_indices)) for i in range(self.n_agent)]
            
        for i in range(self.n_agent):
            self.trans_buffer_ls[i].add_transition(obs[i], actions[i],
                                                  rewards[i], values[i], 
                                                  log_probs[i], done, 
                                                  spillbacks[i])

    def _detect_spillbacks(self, neighbors_info):
        """æ£æµé»å±è·¯å£ä¹é´æ¯å¦åçæº¢åº
        
        åæ°:
        neighbors_info: åå«æ¯ä¸ªæºè½ä½é»å±ç¶æçå­å¸åè¡¨ï¼åå«æ¯å¦æspillbackçæ å¿
        
        è¿å:
        spillbacks: åè¡¨ï¼è¡¨ç¤ºæ¯ä¸ªæºè½ä½çé»å±è·¯å£æ¯å¦åçæº¢åºï¼äºåå¼ï¼
        """
        spillbacks = []
        
        for i in range(self.n_agent):
            # è·åå½åæºè½ä½çé»å±ç´¢å¼
            neighbor_indices = self.policy_ls[i].neighbor_indices
            agent_spillbacks = np.zeros(len(neighbor_indices))
            
            # å¦ææé»å±ä¿¡æ¯
            if neighbors_info[i]:
                for j, neighbor_idx in enumerate(neighbor_indices):
                    # æ£æ¥æ¯å¦æspillbackæ å¿
                    if neighbor_idx in neighbors_info[i]:
                        if neighbors_info[i][neighbor_idx]['has_spillback']:
                            # å¦æææº¢åºï¼è®¾ä¸º1
                            agent_spillbacks[j] = 1.0
                        else:
                            # å¦ææ²¡ææº¢åºï¼è®¾ä¸º0
                            agent_spillbacks[j] = 0.0
            
            spillbacks.append(agent_spillbacks)
        
        return spillbacks

    def _enhance_observations(self, obs, spillbacks, neighbors_info):
        """Enhance observations with spillback information
        
        Parameters:
        obs: List of original observations for all agents
        spillbacks: List of spillback detection results for all agents
        neighbors_info: Neighbor state information
        
        Returns:
        enhanced_obs: List of enhanced observations with spillback information
        """
        enhanced_obs = []
        
        for i in range(self.n_agent):
            # Get the base observation
            base_obs = obs[i].copy()  # Copy to avoid modifying original data
            
            # Get the neighbor indices for current agent
            neighbor_indices = self.policy_ls[i].neighbor_indices
            n_neighbors = len(neighbor_indices)
            
            # Ensure spillbacks[i] has the correct shape (n_neighbors,)
            if len(spillbacks[i]) != n_neighbors:
                spillback_vector = np.zeros(n_neighbors)
                for j in range(min(len(spillbacks[i]), n_neighbors)):
                    spillback_vector[j] = spillbacks[i][j]
            else:
                spillback_vector = spillbacks[i]
            
            # Log dimension info for debugging
            logging.debug("Agent {0}: base_obs shape {1}, spillback shape {2}".format(
                i, base_obs.shape, spillback_vector.shape))
            
            # Concatenate base observation with spillback vector
            enhanced_ob = np.concatenate([base_obs, spillback_vector])
            enhanced_obs.append(enhanced_ob)
            
            # Log the final dimension going into LSTM
            logging.debug("Agent {0}: enhanced observation shape {1}".format(
                i, enhanced_ob.shape))
        
        return enhanced_obs

    def forward(self, obs, done, neighbors_info=None, out_type='pv'):
        """ååä¼ æ­ï¼æ ¹æ®out_typeè¿åä¸åç±»åçè¾åº
        'p': è¿åç­ç¥ï¼å¨ä½æ¦çåå¸ï¼
        'v': è¿åå¼å½æ°
        'pv': è¿åç­ç¥åå¼å½æ°
        'pvlogp': è¿åç­ç¥ãå¼å½æ°ålogæ¦ç
        
        neighbors_info: ä¸ä¸ªå­å¸ï¼åå«æ¯ä¸ªæºè½ä½é»å±çä¿¡æ¯ï¼ç¨äºæ£æµæº¢åº
        """
        if neighbors_info is None:
            neighbors_info = [{} for _ in range(self.n_agent)]
            
        # å¤çæº¢åºæ£æµé»è¾
        spillbacks = self._detect_spillbacks(neighbors_info)
        
        # å°æº¢åºä¿¡æ¯æ·»å å°è§æµä¸­
        enhanced_obs = self._enhance_observations(obs, spillbacks, neighbors_info)
        
        if out_type == 'pvlogp':
            # è·åç­ç¥ãå¼å½æ°åå¨ä½çlogæ¦ç
            policies, values = [], []
            log_probs = []
            for i in range(self.n_agent):
                pi, v, logp = self.policy_ls[i].forward(self.sess, enhanced_obs[i], done, 'pvlogp')
                policies.append(pi)
                values.append(v)
                log_probs.append(logp)
            return policies, values, log_probs, spillbacks
        else:
            # å¤çå¶ä»è¾åºç±»å
            if 'p' in out_type:
                policies = []
                for i in range(self.n_agent):
                    if 'v' in out_type:
                        pi, v = self.policy_ls[i].forward(self.sess, enhanced_obs[i], done, 'pv')
                        policies.append(pi)
                    else:
                        pi = self.policy_ls[i].forward(self.sess, enhanced_obs[i], done, 'p')
                        policies.append(pi)
                
                if 'v' in out_type:
                    values = []
                    for i in range(self.n_agent):
                        _, v = self.policy_ls[i].forward(self.sess, enhanced_obs[i], done, 'pv')
                        values.append(v)
                    return policies, values, spillbacks
                return policies, spillbacks
            else:  # åªæ'v'
                values = []
                for i in range(self.n_agent):
                    v = self.policy_ls[i].forward(self.sess, enhanced_obs[i], done, 'v')
                    values.append(v)
                return values, spillbacks

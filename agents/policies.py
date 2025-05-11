# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from agents.utils import *
import bisect
import logging

class ACPolicy:
    def __init__(self, n_a, n_s, n_step, policy_name, agent_name):
        self.name = policy_name
        if agent_name is not None:
            # for multi-agent system
            self.name += '_' + str(agent_name)
        self.n_a = n_a
        self.n_s = n_s
        self.n_step = n_step

    def forward(self, ob, *_args, **_kwargs):
        raise NotImplementedError()

    def _build_out_net(self, h, out_type):
        if out_type == 'pi':
            pi = fc(h, out_type, self.n_a, act=tf.nn.softmax)
            return tf.squeeze(pi)
        else:
            v = fc(h, out_type, 1, act=lambda x: x)
            return tf.squeeze(v)

    def _get_forward_outs(self, out_type):
        outs = []
        if 'p' in out_type:
            outs.append(self.pi)
        if 'v' in out_type:
            outs.append(self.v)
        return outs

    def _return_forward_outs(self, out_values):
        if len(out_values) == 1:
            return out_values[0]
        return out_values

    def prepare_loss(self, v_coef, max_grad_norm, alpha, epsilon):
        self.A = tf.placeholder(tf.int32, [self.n_step])
        self.ADV = tf.placeholder(tf.float32, [self.n_step])
        self.R = tf.placeholder(tf.float32, [self.n_step])
        self.entropy_coef = tf.placeholder(tf.float32, [])
        A_sparse = tf.one_hot(self.A, self.n_a)
        log_pi = tf.log(tf.clip_by_value(self.pi, 1e-10, 1.0))
        entropy = -tf.reduce_sum(self.pi * log_pi, axis=1)
        entropy_loss = -tf.reduce_mean(entropy) * self.entropy_coef
        policy_loss = -tf.reduce_mean(tf.reduce_sum(log_pi * A_sparse, axis=1) * self.ADV)
        value_loss = tf.reduce_mean(tf.square(self.R - self.v)) * 0.5 * v_coef
        self.loss = policy_loss + value_loss + entropy_loss

        wts = tf.trainable_variables(scope=self.name)
        grads = tf.gradients(self.loss, wts)
        if max_grad_norm > 0:
            grads, self.grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        self.lr = tf.placeholder(tf.float32, [])
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr, decay=alpha,
                                                   epsilon=epsilon)
        self._train = self.optimizer.apply_gradients(list(zip(grads, wts)))
        # monitor training
        if self.name.endswith('_0a'):
            summaries = []
            # summaries.append(tf.summary.scalar('loss/%s_entropy_loss' % self.name, entropy_loss))
            summaries.append(tf.summary.scalar('loss/%s_policy_loss' % self.name, policy_loss))
            summaries.append(tf.summary.scalar('loss/%s_value_loss' % self.name, value_loss))
            summaries.append(tf.summary.scalar('loss/%s_total_loss' % self.name, self.loss))
            # summaries.append(tf.summary.scalar('train/%s_lr' % self.name, self.lr))
            # summaries.append(tf.summary.scalar('train/%s_entropy_beta' % self.name, self.entropy_coef))
            summaries.append(tf.summary.scalar('train/%s_gradnorm' % self.name, self.grad_norm))
            self.summary = tf.summary.merge(summaries)


class LstmACPolicy(ACPolicy):
    def __init__(self, n_s, n_a, n_w, n_step, n_fc_wave=128, n_fc_wait=32, n_lstm=64, name=None):
        super().__init__(n_a, n_s, n_step, 'lstm', name)
        self.n_lstm = n_lstm
        self.n_fc_wait = n_fc_wait
        self.n_fc_wave = n_fc_wave
        self.n_w = n_w
        self.ob_fw = tf.placeholder(tf.float32, [1, n_s + n_w]) # forward 1-step
        self.done_fw = tf.placeholder(tf.float32, [1])
        self.ob_bw = tf.placeholder(tf.float32, [n_step, n_s + n_w]) # backward n-step
        self.done_bw = tf.placeholder(tf.float32, [n_step])
        self.states = tf.placeholder(tf.float32, [2, n_lstm * 2])
        with tf.variable_scope(self.name):
            # pi and v use separate nets
            self.pi_fw, pi_state = self._build_net('forward', 'pi')
            self.v_fw, v_state = self._build_net('forward', 'v')
            pi_state = tf.expand_dims(pi_state, 0)
            v_state = tf.expand_dims(v_state, 0)
            self.new_states = tf.concat([pi_state, v_state], 0)
        with tf.variable_scope(self.name, reuse=True):
            self.pi, _ = self._build_net('backward', 'pi')
            self.v, _ = self._build_net('backward', 'v')
        self._reset()

    def _build_net(self, in_type, out_type):
        if in_type == 'forward':
            ob = self.ob_fw
            done = self.done_fw
        else:
            ob = self.ob_bw
            done = self.done_bw
        if out_type == 'pi':
            states = self.states[0]
        else:
            states = self.states[1]
        if self.n_w == 0:
            h = fc(ob, out_type + '_fcw', self.n_fc_wave)
        else:
            h0 = fc(ob[:, :self.n_s], out_type + '_fcw', self.n_fc_wave)
            h1 = fc(ob[:, self.n_s:], out_type + '_fct', self.n_fc_wait)
            h = tf.concat([h0, h1], 1)
        h, new_states = lstm(h, done, states, out_type + '_lstm')
        out_val = self._build_out_net(h, out_type)
        return out_val, new_states

    def _reset(self):
        # forget the cumulative states every cum_step
        self.states_fw = np.zeros((2, self.n_lstm * 2), dtype=np.float32)
        self.states_bw = np.zeros((2, self.n_lstm * 2), dtype=np.float32)

    def forward(self, sess, ob, done, out_type='pv'):
        outs = self._get_forward_outs(out_type)
        # update state only when p is called
        if 'p' in out_type:
            outs.append(self.new_states)
        out_values = sess.run(outs, {self.ob_fw:np.array([ob]),
                                     self.done_fw:np.array([done]),
                                     self.states:self.states_fw})
        if 'p' in out_type:
            self.states_fw = out_values[-1]
            out_values = out_values[:-1]
        return self._return_forward_outs(out_values)

    def backward(self, sess, obs, acts, dones, Rs, Advs, cur_lr, cur_beta,
                 summary_writer=None, global_step=None):
        if summary_writer is None:
            ops = self._train
        else:
            ops = [self.summary, self._train]
        outs = sess.run(ops,
                        {self.ob_bw: obs,
                         self.done_bw: dones,
                         self.states: self.states_bw,
                         self.A: acts,
                         self.ADV: Advs,
                         self.R: Rs,
                         self.lr: cur_lr,
                         self.entropy_coef: cur_beta})
        self.states_bw = np.copy(self.states_fw)
        if summary_writer is not None:
            summary_writer.add_summary(outs[0], global_step=global_step)

    def _get_forward_outs(self, out_type):
        outs = []
        if 'p' in out_type:
            outs.append(self.pi_fw)
        if 'v' in out_type:
            outs.append(self.v_fw)
        return outs


class FPLstmACPolicy(LstmACPolicy):
    def __init__(self, n_s, n_a, n_w, n_f, n_step, n_fc_wave=128, n_fc_wait=32, n_fc_fp=32, n_lstm=64, name=None):
        ACPolicy.__init__(self, n_a, n_s, n_step, 'fplstm', name)
        self.n_lstm = n_lstm
        self.n_fc_wave = n_fc_wave
        self.n_fc_wait = n_fc_wait
        self.n_fc_fp = n_fc_fp
        self.n_w = n_w
        self.ob_fw = tf.placeholder(tf.float32, [1, n_s + n_w + n_f]) # forward 1-step
        self.done_fw = tf.placeholder(tf.float32, [1])
        self.ob_bw = tf.placeholder(tf.float32, [n_step, n_s + n_w + n_f]) # backward n-step
        self.done_bw = tf.placeholder(tf.float32, [n_step])
        self.states = tf.placeholder(tf.float32, [2, n_lstm * 2])
        with tf.variable_scope(self.name):
            # pi and v use separate nets
            self.pi_fw, pi_state = self._build_net('forward', 'pi')
            self.v_fw, v_state = self._build_net('forward', 'v')
            pi_state = tf.expand_dims(pi_state, 0)
            v_state = tf.expand_dims(v_state, 0)
            self.new_states = tf.concat([pi_state, v_state], 0)
        with tf.variable_scope(self.name, reuse=True):
            self.pi, _ = self._build_net('backward', 'pi')
            self.v, _ = self._build_net('backward', 'v')
        self._reset()

    def _build_net(self, in_type, out_type):
        if in_type == 'forward':
            ob = self.ob_fw
            done = self.done_fw
        else:
            ob = self.ob_bw
            done = self.done_bw
        if out_type == 'pi':
            states = self.states[0]
        else:
            states = self.states[1]
        h0 = fc(ob[:, :self.n_s], out_type + '_fcw', self.n_fc_wave)
        h1 = fc(ob[:, (self.n_s + self.n_w):], out_type + '_fcf', self.n_fc_fp)
        if self.n_w == 0:
            h = tf.concat([h0, h1], 1)
        else:
            h2 = fc(ob[:, self.n_s: (self.n_s + self.n_w)], out_type + '_fct', self.n_fc_wait)
            h = tf.concat([h0, h1, h2], 1)
        h, new_states = lstm(h, done, states, out_type + '_lstm')
        out_val = self._build_out_net(h, out_type)
        return out_val, new_states


class FcACPolicy(ACPolicy):
    def __init__(self, n_s, n_a, n_w, n_step, n_fc_wave=128, n_fc_wait=32, n_lstm=64, name=None):
        super().__init__(n_a, n_s, n_step, 'fc', name)
        self.n_fc_wave = n_fc_wave
        self.n_fc_wait = n_fc_wait
        self.n_fc = n_lstm
        self.n_w = n_w
        self.obs = tf.placeholder(tf.float32, [None, n_s + n_w])
        with tf.variable_scope(self.name):
            # pi and v use separate nets
            self.pi = self._build_net('pi')
            self.v = self._build_net('v')

    def _build_net(self, out_type):
        if self.n_w == 0:
            h = fc(self.obs, out_type + '_fcw', self.n_fc_wave)
        else:
            h0 = fc(self.obs[:, :self.n_s], out_type + '_fcw', self.n_fc_wave)
            h1 = fc(self.obs[:, self.n_s:], out_type + '_fct', self.n_fc_wait)
            h = tf.concat([h0, h1], 1)
        h = fc(h, out_type + '_fc', self.n_fc)
        return self._build_out_net(h, out_type)

    def forward(self, sess, ob, done, out_type='pv'):
        outs = self._get_forward_outs(out_type)
        out_values = sess.run(outs, {self.obs: np.array([ob])})
        return self._return_forward_outs(out_values)

    def backward(self, sess, obs, acts, dones, Rs, Advs, cur_lr, cur_beta,
                 summary_writer=None, global_step=None):
        if summary_writer is None:
            ops = self._train
        else:
            ops = [self.summary, self._train]
        outs = sess.run(ops,
                        {self.obs: obs,
                         self.A: acts,
                         self.ADV: Advs,
                         self.R: Rs,
                         self.lr: cur_lr,
                         self.entropy_coef: cur_beta})
        if summary_writer is not None:
            summary_writer.add_summary(outs[0], global_step=global_step)


class FPFcACPolicy(FcACPolicy):
    def __init__(self, n_s, n_a, n_w, n_f, n_step, n_fc_wave=128, n_fc_wait=32, n_fc_fp=32, n_lstm=64, name=None):
        ACPolicy.__init__(self, n_a, n_s, n_step, 'fpfc', name)
        self.n_fc_wave = n_fc_wave
        self.n_fc_wait = n_fc_wait
        self.n_fc_fp = n_fc_fp
        self.n_fc = n_lstm
        self.n_w = n_w
        self.obs = tf.placeholder(tf.float32, [None, n_s + n_w + n_f])
        with tf.variable_scope(self.name):
            # pi and v use separate nets
            self.pi = self._build_net('pi')
            self.v = self._build_net('v')

    def _build_net(self, out_type):
        h0 = fc(ob[:, :self.n_s], out_type + '_fcw', self.n_fc_wave)
        h1 = fc(ob[:, (self.n_s + self.n_w):], out_type + '_fcf', self.n_fc_fp)
        if self.n_w == 0:
            h = tf.concat([h0, h1], 1)
        else:
            h2 = fc(ob[:, self.n_s: (self.n_s + self.n_w)], out_type + '_fct', self.n_fc_wait)
            h = tf.concat([h0, h1, h2], 1)
        h = fc(h, out_type + '_fc', self.n_fc)
        return self._build_out_net(h, out_type)


class QPolicy:
    def __init__(self, n_a, n_s, n_step, policy_name, agent_name):
        self.name = policy_name
        if agent_name is not None:
            # for multi-agent system
            self.name += '_' + str(agent_name)
        self.n_a = n_a
        self.n_s = n_s
        self.n_step = n_step

    def forward(self, ob, *_args, **_kwargs):
        raise NotImplementedError()

    def _build_fc_net(self, h, n_fc_ls):
        for i, n_fc in enumerate(n_fc_ls):
            h = fc(h, 'q_fc_%d' % i, n_fc)
        q = fc(h, 'q', self.n_a, act=lambda x: x)
        return tf.squeeze(q)

    def _build_net(self):
        raise NotImplementedError()

    def prepare_loss(self, max_grad_norm, gamma):
        self.A = tf.placeholder(tf.int32, [self.n_step])
        self.S1 = tf.placeholder(tf.float32, [self.n_step, self.n_s + self.n_w])
        self.R = tf.placeholder(tf.float32, [self.n_step])
        self.DONE = tf.placeholder(tf.bool, [self.n_step])
        A_sparse = tf.one_hot(self.A, self.n_a)

        # backward
        with tf.variable_scope(self.name + '_q', reuse=True):
            q0s = self._build_net(self.S)
            q0 = tf.reduce_sum(q0s * A_sparse, axis=1)
        with tf.variable_scope(self.name + '_q', reuse=True):
            q1s = self._build_net(self.S1)
            q1 = tf.reduce_max(q1s, axis=1)
        tq = tf.stop_gradient(tf.where(self.DONE, self.R, self.R + gamma * q1))
        self.loss = tf.reduce_mean(tf.square(q0 - tq))

        wts = tf.trainable_variables(scope=self.name)
        grads = tf.gradients(self.loss, wts)
        if max_grad_norm > 0:
            grads, self.grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        self.lr = tf.placeholder(tf.float32, [])
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self._train = self.optimizer.apply_gradients(list(zip(grads, wts)))
        # monitor training
        if self.name.endswith('_0a'):
            summaries = []
            summaries.append(tf.summary.scalar('train/%s_loss' % self.name, self.loss))
            summaries.append(tf.summary.scalar('train/%s_q' % self.name, tf.reduce_mean(q0)))
            summaries.append(tf.summary.scalar('train/%s_tq' % self.name, tf.reduce_mean(tq)))
            summaries.append(tf.summary.scalar('train/%s_gradnorm' % self.name, self.grad_norm))
            self.summary = tf.summary.merge(summaries)


class DeepQPolicy(QPolicy):
    def __init__(self, n_s, n_a, n_w, n_step, n_fc0=128, n_fc=64, name=None):
        super().__init__(n_a, n_s, n_step, 'dqn', name)
        self.n_fc = n_fc
        self.n_fc0 = n_fc0
        self.n_w = n_w
        self.S = tf.placeholder(tf.float32, [None, n_s + n_w])
        with tf.variable_scope(self.name + '_q'):
            self.qvalues = self._build_net(self.S)

    def _build_net(self, S):
        if self.n_w == 0:
            h = fc(S, 'q_fcw', self.n_fc0)
        else:
            h0 = fc(S[:, :self.n_s], 'q_fcw', self.n_fc0)
            h1 = fc(S[:, self.n_s:], 'q_fct', self.n_fc0 / 4)
            h = tf.concat([h0, h1], 1)
        return self._build_fc_net(h, [self.n_fc])

    def forward(self, sess, ob):
        return sess.run(self.qvalues, {self.S: np.array([ob])})

    def backward(self, sess, obs, acts, next_obs, dones, rs, cur_lr,
                 summary_writer=None, global_step=None):
        if summary_writer is None:
            ops = self._train
        else:
            ops = [self.summary, self._train]
        outs = sess.run(ops,
                        {self.S: obs,
                         self.A: acts,
                         self.S1: next_obs,
                         self.DONE: dones,
                         self.R: rs,
                         self.lr: cur_lr})
        if summary_writer is not None:
            summary_writer.add_summary(outs[0], global_step=global_step)


class LRQPolicy(DeepQPolicy):
    def __init__(self, n_s, n_a, n_step, name=None):
        QPolicy.__init__(self, n_a, n_s, n_step, 'lr', name)
        self.S = tf.placeholder(tf.float32, [None, n_s])
        self.n_w = 0
        with tf.variable_scope(self.name + '_q'):
            self.qvalues = self._build_net(self.S)

    def _build_net(self, S):
        return self._build_fc_net(S, [])


class PPOPolicy(LstmACPolicy):
    def __init__(self, n_s, n_a, n_w, n_step, n_fc_wave=128, n_fc_wait=32, n_lstm=64, name=None):
        super().__init__(n_s, n_a, n_w, n_step, n_fc_wave, n_fc_wait, n_lstm, name)
        self.old_log_probs = tf.placeholder(tf.float32, [self.n_step])
        
    def prepare_loss(self, v_coef, max_grad_norm, alpha, epsilon, clip_ratio, entropy_coef):
        
        self.A = tf.placeholder(tf.int32, [self.n_step])
        self.ADV = tf.placeholder(tf.float32, [self.n_step])
        self.R = tf.placeholder(tf.float32, [self.n_step])
        self.entropy_coef = tf.placeholder(tf.float32, [])
        self.clip_ratio = clip_ratio
        
        
        A_sparse = tf.one_hot(self.A, self.n_a)
        log_pi = tf.log(tf.clip_by_value(self.pi, 1e-10, 1.0))
        
        
        curr_log_probs = tf.reduce_sum(log_pi * A_sparse, axis=1)
        
        
        ratio = tf.exp(curr_log_probs - self.old_log_probs)
        
       
        clip_adv = tf.clip_by_value(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * self.ADV
        policy_loss = -tf.reduce_mean(tf.minimum(ratio * self.ADV, clip_adv))
        
        
        entropy = -tf.reduce_sum(self.pi * log_pi, axis=1)
        entropy_loss = -tf.reduce_mean(entropy) * self.entropy_coef
        value_loss = tf.reduce_mean(tf.square(self.R - self.v)) * 0.5 * v_coef
        
       
        self.loss = policy_loss + value_loss + entropy_loss
        
        
        wts = tf.trainable_variables(scope=self.name)
        grads = tf.gradients(self.loss, wts)
        if max_grad_norm > 0:
            grads, self.grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        self.lr = tf.placeholder(tf.float32, [])
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr, decay=alpha,
                                                  epsilon=epsilon)
        self._train = self.optimizer.apply_gradients(list(zip(grads, wts)))
        
       
        if self.name.endswith('_0a'):
            summaries = []
            summaries.append(tf.summary.scalar('loss/%s_policy_loss' % self.name, policy_loss))
            summaries.append(tf.summary.scalar('loss/%s_value_loss' % self.name, value_loss))
            summaries.append(tf.summary.scalar('loss/%s_entropy_loss' % self.name, entropy_loss))
            summaries.append(tf.summary.scalar('loss/%s_total_loss' % self.name, self.loss))
            summaries.append(tf.summary.scalar('train/%s_gradnorm' % self.name, self.grad_norm))
            self.summary = tf.summary.merge(summaries)
    
    def _get_forward_outs(self, out_type):
        
        outs = []
        if 'p' in out_type:
            outs.append(self.pi_fw)
        if 'v' in out_type:
            outs.append(self.v_fw)
        if 'logp' in out_type:
            
            a_sparse = tf.one_hot(tf.range(self.n_a), self.n_a)
            log_pi = tf.log(tf.clip_by_value(self.pi_fw, 1e-10, 1.0))
            outs.append(log_pi)
        return outs
    
    def forward(self, sess, ob, done, out_type='pv'):
       
        outs = self._get_forward_outs(out_type)
        
        if 'p' in out_type:
            outs.append(self.new_states)
        out_values = sess.run(outs, {self.ob_fw: np.array([ob]),
                                    self.done_fw: np.array([done]),
                                    self.states: self.states_fw})
        if 'p' in out_type:
            self.states_fw = out_values[-1]
            out_values = out_values[:-1]
        return self._return_forward_outs(out_values)
    
    def backward(self, sess, obs, acts, dones, Rs, Advs, old_log_probs, cur_lr,
                summary_writer=None, global_step=None):
       
        if summary_writer is None:
            ops = self._train
        else:
            ops = [self.summary, self._train]
        outs = sess.run(ops,
                       {self.ob_bw: obs,
                        self.done_bw: dones,
                        self.states: self.states_bw,
                        self.A: acts,
                        self.ADV: Advs,
                        self.R: Rs,
                        self.old_log_probs: old_log_probs,
                        self.lr: cur_lr,
                        self.entropy_coef: self.entropy_coef})
        self.states_bw = np.copy(self.states_fw)
        if summary_writer is not None:
            summary_writer.add_summary(outs[0], global_step=global_step)
       
        
class SPPOPolicy(LstmACPolicy):
    """Spillback-aware PPO policy network.
    Inherits from LstmACPolicy directly to avoid initialization order issues.
    Adds the capability to detect and handle spillbacks between neighboring intersections.
    """
    def __init__(self, n_s, n_a, n_w, n_step, n_fc_wave=128, n_fc_wait=32, n_lstm=64, name=None):
        # Parse neighbor relationships first
        self.neighbor_indices = self._parse_neighbors(name)
        
        # Calculate spillback information dimension
        n_spillback = len(self.neighbor_indices)
        
        # Set spillback related attributes before any super init
        self.spillback_start_idx = n_s + n_w
        self.orig_n_s = n_s
        self.orig_n_w = n_w
        
        # Instead of using super().__init__, directly initialize from ACPolicy
        # This bypasses the intermediate classes' initialization logic
        ACPolicy.__init__(self, n_a, n_s, n_step, 'sppo', name)
        
        # Initialize PPO-specific placeholder
        self.old_log_probs = tf.placeholder(tf.float32, [n_step])
        
        # Setup network parameters
        self.n_lstm = n_lstm
        self.n_fc_wait = n_fc_wait
        self.n_fc_wave = n_fc_wave
        self.n_w = n_w
        
        # Define placeholders with spillback dimension
        self.ob_fw = tf.placeholder(tf.float32, [1, n_s + n_w + n_spillback])
        self.done_fw = tf.placeholder(tf.float32, [1])
        self.ob_bw = tf.placeholder(tf.float32, [n_step, n_s + n_w + n_spillback])
        self.done_bw = tf.placeholder(tf.float32, [n_step])
        self.states = tf.placeholder(tf.float32, [2, n_lstm * 2])
        
        # Build network structure
        with tf.variable_scope(self.name):
            self.pi_fw, pi_state = self._build_net('forward', 'pi')
            self.v_fw, v_state = self._build_net('forward', 'v')
            pi_state = tf.expand_dims(pi_state, 0)
            v_state = tf.expand_dims(v_state, 0)
            self.new_states = tf.concat([pi_state, v_state], 0)
        
        with tf.variable_scope(self.name, reuse=True):
            self.pi, _ = self._build_net('backward', 'pi')
            self.v, _ = self._build_net('backward', 'v')
        
        # Initialize LSTM states
        self._reset()
        
        logging.info("SPPOPolicy initialized: n_s={0}, n_w={1}, spillback_dim={2}".format(
        n_s, n_w, len(self.neighbor_indices)))
        logging.info("LSTM state dimensions: {0}".format(self.n_lstm * 2))
        logging.info("Input shape: {0}, LSTM hidden units: {1}".format(
            n_s + n_w + len(self.neighbor_indices), n_lstm))
        logging.info("Agent {0}: Total observation dimension: {1}".format(
            name, n_s + n_w + len(self.neighbor_indices)))
        
    def _parse_neighbors(self, agent_name):
        """Parse agent's neighbor node indices
        
        Based on SUMO network node relationships, determine the current node's neighbors.
        Simplified to fixed neighbor relationships, in real applications should read from XML config.
        
        Parameters:
        agent_name: Agent name, like "0a", "1a", etc.
        
        Returns:
        neighbor_indices: List of neighbor node indices
        """
        # Get index from agent name
        if agent_name is None:
            return []
            
        try:
            agent_idx = int(agent_name.split('a')[0])
        except:
            return []
            
        # Parse large_grid neighbor relationships directly
        # For 5x5 grid, based on node configuration in exp.nod.xml
        # Node IDs are nt1 to nt25, corresponding to agent indices 0 to 24
        neighbors = []
        
        # Grid size
        grid_size = 5
        
        # Calculate grid row and column
        row = agent_idx // grid_size
        col = agent_idx % grid_size
        
        # Check neighbor above
        if row > 0:
            neighbors.append((row - 1) * grid_size + col)
        
        # Check neighbor below
        if row < grid_size - 1:
            neighbors.append((row + 1) * grid_size + col)
        
        # Check neighbor to the left
        if col > 0:
            neighbors.append(row * grid_size + col - 1)
        
        # Check neighbor to the right
        if col < grid_size - 1:
            neighbors.append(row * grid_size + col + 1)
            
        return neighbors
    
    def _build_net(self, in_type, out_type):
        """Build network with spillback information processing
        
        Completely override the parent's _build_net method to ensure
        spillback_start_idx is used correctly
        """
        if in_type == 'forward':
            ob = self.ob_fw
            done = self.done_fw
        else:
            ob = self.ob_bw
            done = self.done_bw
            
        if out_type == 'pi':
            states = self.states[0]
        else:
            states = self.states[1]
            
        # Separate wave, wait and spillback information from observation
        wave_obs = ob[:, :self.n_s]
        if self.n_w > 0:
            wait_obs = ob[:, self.n_s:self.spillback_start_idx]
            spillback_obs = ob[:, self.spillback_start_idx:]
            
            # Process basic observation
            h0 = fc(wave_obs, out_type + '_fcw', self.n_fc_wave)
            h1 = fc(wait_obs, out_type + '_fct', self.n_fc_wait)
            
            # Process spillback information
            h2 = fc(spillback_obs, out_type + '_fcs1', self.n_fc_wait)
            h2 = fc(h2, out_type + '_fcs2', int(self.n_fc_wait / 2))
            
            # Merge all features
            h = tf.concat([h0, h1, h2], 1)
        else:
            spillback_obs = ob[:, self.n_s:]
            
            # Process basic observation
            h0 = fc(wave_obs, out_type + '_fcw', self.n_fc_wave)
            
            # Process spillback information
            h1 = fc(spillback_obs, out_type + '_fcs1', self.n_fc_wait)
            h1 = fc(h1, out_type + '_fcs2', int(self.n_fc_wait / 2))
            
            # Merge all features
            h = tf.concat([h0, h1], 1)
            
        # LSTM processing
        h, new_states = lstm(h, done, states, out_type + '_lstm')
        
        # Output layer
        out_val = self._build_out_net(h, out_type)
        return out_val, new_states
    
    def _reset(self):
        logging.info("Resetting LSTM states with n_lstm={0}".format(self.n_lstm))
        self.states_fw = np.zeros((2, self.n_lstm * 2), dtype=np.float32)
        self.states_bw = np.zeros((2, self.n_lstm * 2), dtype=np.float32)
    
    def _get_forward_outs(self, out_type):
        """Get outputs for forward pass including log probabilities if needed"""
        outs = []
        if 'p' in out_type:
            outs.append(self.pi_fw)
        if 'v' in out_type:
            outs.append(self.v_fw)
        if 'logp' in out_type:
            # Current policy's log probabilities
            log_pi = tf.log(tf.clip_by_value(self.pi_fw, 1e-10, 1.0))
            outs.append(log_pi)
        return outs
    
    def prepare_loss(self, v_coef, max_grad_norm, alpha, epsilon, clip_ratio, entropy_coef):
        """Prepare PPO loss function with clipping"""
        self.A = tf.placeholder(tf.int32, [self.n_step])
        self.ADV = tf.placeholder(tf.float32, [self.n_step])
        self.R = tf.placeholder(tf.float32, [self.n_step])
        self.entropy_coef = tf.placeholder(tf.float32, [])
        self.clip_ratio = clip_ratio
        
        # Build PPO loss function
        A_sparse = tf.one_hot(self.A, self.n_a)
        log_pi = tf.log(tf.clip_by_value(self.pi, 1e-10, 1.0))
        
        # Calculate current action's log probabilities
        curr_log_probs = tf.reduce_sum(log_pi * A_sparse, axis=1)
        
        # Calculate probability ratio
        ratio = tf.exp(curr_log_probs - self.old_log_probs)
        
        # Clipped objective function
        clip_adv = tf.clip_by_value(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * self.ADV
        policy_loss = -tf.reduce_mean(tf.minimum(ratio * self.ADV, clip_adv))
        
        # Entropy and value loss same as A2C
        entropy = -tf.reduce_sum(self.pi * log_pi, axis=1)
        entropy_loss = -tf.reduce_mean(entropy) * self.entropy_coef
        value_loss = tf.reduce_mean(tf.square(self.R - self.v)) * 0.5 * v_coef
        
        # Total loss
        self.loss = policy_loss + value_loss + entropy_loss
        
        # Parameter updates
        wts = tf.trainable_variables(scope=self.name)
        grads = tf.gradients(self.loss, wts)
        if max_grad_norm > 0:
            grads, self.grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        self.lr = tf.placeholder(tf.float32, [])
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr, decay=alpha,
                                                  epsilon=epsilon)
        self._train = self.optimizer.apply_gradients(list(zip(grads, wts)))
        
        # Monitor training
        if self.name.endswith('_0a'):
            summaries = []
            summaries.append(tf.summary.scalar('loss/%s_policy_loss' % self.name, policy_loss))
            summaries.append(tf.summary.scalar('loss/%s_value_loss' % self.name, value_loss))
            summaries.append(tf.summary.scalar('loss/%s_entropy_loss' % self.name, entropy_loss))
            summaries.append(tf.summary.scalar('loss/%s_total_loss' % self.name, self.loss))
            summaries.append(tf.summary.scalar('train/%s_gradnorm' % self.name, self.grad_norm))
            self.summary = tf.summary.merge(summaries)
    
    def forward(self, sess, ob, done, out_type='pv'):
        """Forward pass implementation for SPPO"""
        outs = self._get_forward_outs(out_type)
        # Update state only when p is called
        if 'p' in out_type:
            outs.append(self.new_states)
        out_values = sess.run(outs, {self.ob_fw: np.array([ob]),
                                    self.done_fw: np.array([done]),
                                    self.states: self.states_fw})
        if 'p' in out_type:
            self.states_fw = out_values[-1]
            out_values = out_values[:-1]
        return self._return_forward_outs(out_values)
    
    def backward(self, sess, obs, acts, dones, Rs, Advs, old_log_probs, cur_lr,
                summary_writer=None, global_step=None):
        """Backward pass implementation for SPPO"""
        if summary_writer is None:
            ops = self._train
        else:
            ops = [self.summary, self._train]
        outs = sess.run(ops,
                       {self.ob_bw: obs,
                        self.done_bw: dones,
                        self.states: self.states_bw,
                        self.A: acts,
                        self.ADV: Advs,
                        self.R: Rs,
                        self.old_log_probs: old_log_probs,
                        self.lr: cur_lr,
                        self.entropy_coef: self.entropy_coef})
        self.states_bw = np.copy(self.states_fw)
        if summary_writer is not None:
            summary_writer.add_summary(outs[0], global_step=global_step)

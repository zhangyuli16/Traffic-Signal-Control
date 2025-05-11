# -*- coding: utf-8 -*-
import numpy as np
import random
import tensorflow as tf

"""
initializers
"""
DEFAULT_SCALE = np.sqrt(2)
DEFAULT_MODE = 'fan_in'

def ortho_init(scale=DEFAULT_SCALE, mode=None):
    def _ortho_init(shape, dtype, partition_info=None):
        # lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2: # fc: in, out
            flat_shape = shape
        elif (len(shape) == 3) or (len(shape) == 4): # 1d/2dcnn: (in_h), in_w, in_c, out
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        a = np.random.standard_normal(flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q).astype(np.float32)
    return _ortho_init


def norm_init(scale=DEFAULT_SCALE, mode=DEFAULT_MODE):
    def _norm_init(shape, dtype, partition_info=None):
        shape = tuple(shape)
        if len(shape) == 2:
            n_in = shape[0]
        elif (len(shape) == 3) or (len(shape) == 4):
            n_in = np.prod(shape[:-1])
        a = np.random.standard_normal(shape)
        if mode == 'fan_in':
            n = n_in
        elif mode == 'fan_out':
            n = shape[-1]
        elif mode == 'fan_avg':
            n = 0.5 * (n_in + shape[-1])
        return (scale * a / np.sqrt(n)).astype(np.float32)

DEFAULT_METHOD = ortho_init
"""
layers
"""
def conv(x, scope, n_out, f_size, stride=1, pad='VALID', f_size_w=None, act=tf.nn.relu,
         conv_dim=1, init_scale=DEFAULT_SCALE, init_mode=None, init_method=DEFAULT_METHOD):
    with tf.variable_scope(scope):
        b = tf.get_variable("b", [n_out], initializer=tf.constant_initializer(0.0))
        if conv_dim == 1:
            n_c = x.shape[2].value
            w = tf.get_variable("w", [f_size, n_c, n_out],
                                initializer=init_method(init_scale, init_mode))
            z = tf.nn.conv1d(x, w, stride=stride, padding=pad) + b
        elif conv_dim == 2:
            n_c = x.shape[3].value
            if f_size_w is None:
                f_size_w = f_size
            w = tf.get_variable("w", [f_size, f_size_w, n_c, n_out],
                                initializer=init_method(init_scale, init_mode))
            z = tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding=pad) + b
        return act(z)


def fc(x, scope, n_out, act=tf.nn.relu, init_scale=DEFAULT_SCALE,
       init_mode=DEFAULT_MODE, init_method=DEFAULT_METHOD):
    with tf.variable_scope(scope):
        n_in = x.shape[1].value
        w = tf.get_variable("w", [n_in, n_out],
                            initializer=init_method(init_scale, init_mode))
        b = tf.get_variable("b", [n_out], initializer=tf.constant_initializer(0.0))
        z = tf.matmul(x, w) + b
        return act(z)


def batch_to_seq(x):
    n_step = x.shape[0].value
    if len(x.shape) == 1:
        x = tf.expand_dims(x, -1)
    return tf.split(axis=0, num_or_size_splits=n_step, value=x)


def seq_to_batch(x):
    return tf.concat(axis=0, values=x)


def lstm(xs, dones, s, scope, init_scale=DEFAULT_SCALE, init_mode=DEFAULT_MODE,
         init_method=DEFAULT_METHOD):
    xs = batch_to_seq(xs)
    # need dones to reset states
    dones = batch_to_seq(dones)
    n_in = xs[0].shape[1].value
    n_out = s.shape[0] // 2
    with tf.variable_scope(scope):
        wx = tf.get_variable("wx", [n_in, n_out*4],
                             initializer=init_method(init_scale, init_mode))
        wh = tf.get_variable("wh", [n_out, n_out*4],
                             initializer=init_method(init_scale, init_mode))
        b = tf.get_variable("b", [n_out*4], initializer=tf.constant_initializer(0.0))
    s = tf.expand_dims(s, 0)
    c, h = tf.split(axis=1, num_or_size_splits=2, value=s)
    for ind, (x, done) in enumerate(zip(xs, dones)):
        c = c * (1-done)
        h = h * (1-done)
        z = tf.matmul(x, wx) + tf.matmul(h, wh) + b
        i, f, o, u = tf.split(axis=1, num_or_size_splits=4, value=z)
        i = tf.nn.sigmoid(i)
        f = tf.nn.sigmoid(f)
        o = tf.nn.sigmoid(o)
        u = tf.tanh(u)
        c = f*c + i*u
        h = o*tf.tanh(c)
        xs[ind] = h
    s = tf.concat(axis=1, values=[c, h])
    return seq_to_batch(xs), tf.squeeze(s)


def test_layers():
    print(tf.__version__)
    tf.reset_default_graph()
    sess = tf.Session()
    n_step = 5
    fc_x = tf.placeholder(tf.float32, [None, 10])
    lstm_x = tf.placeholder(tf.float32, [n_step, 2])
    lstm_done = tf.placeholder(tf.float32, [n_step])
    lstm_s = tf.placeholder(tf.float32, [20])
    conv1_x = tf.placeholder(tf.float32, [None, 8, 1])
    conv2_x = tf.placeholder(tf.float32, [None, 8, 8, 1])
    fc_out = fc(fc_x, 'fc', 10)
    lstm_out, lstm_ns = lstm(lstm_x, lstm_done, lstm_s, 'lstm')
    conv1_out = conv(conv1_x, 'conv1', 10, 4, conv_dim=1)
    conv2_out = conv(conv2_x, 'conv2', 10, 4, conv_dim=2)
    sess.run(tf.global_variables_initializer())
    inputs = {'fc': {fc_x:np.random.randn(n_step, 10)},
              'lstm_done': {lstm_x:np.zeros((n_step, 2)),
                            lstm_done:np.ones(n_step),
                            lstm_s:np.random.randn(20)},
              'lstm': {lstm_x:np.random.randn(n_step, 2),
                       lstm_done:np.zeros(n_step),
                       lstm_s:np.random.randn(20)},
              'conv1': {conv1_x:np.random.randn(n_step, 8, 1)},
              'conv2': {conv2_x:np.random.randn(n_step, 8, 8, 1)}}
    outputs = {'fc': [fc_out], 'lstm_done': [lstm_out, lstm_ns],
               'conv1': [conv1_out], 'conv2': [conv2_out],
               'lstm': [lstm_out, lstm_ns]}
    for scope in ['fc', 'lstm', 'conv1', 'conv2']:
        print(scope)
        wts = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        for wt in wts:
            wt_val = wt.eval(sess)
            print(wt_val.shape)
            print(np.mean(wt_val), np.std(wt_val), np.min(wt_val), np.max(wt_val))
    print('=====================================')
    for x_name in inputs:
        print(x_name)
        out = sess.run(outputs[x_name], inputs[x_name])
        if x_name.startswith('lstm'):
            print(out[0])
            print(out[1])
        else:
            print(out[0].shape)

"""
buffers
"""
class TransBuffer:
    def reset(self):
        self.buffer = []

    @property
    def size(self):
        return len(self.buffer)

    def add_transition(self, ob, a, r, *_args, **_kwargs):
        raise NotImplementedError()

    def sample_transition(self, *_args, **_kwargs):
        raise NotImplementedError()


class OnPolicyBuffer(TransBuffer):
    def __init__(self, gamma):
        self.gamma = gamma
        self.reset()

    def reset(self, done=False):
        # the done before each step is required
        self.obs = []
        self.acts = []
        self.rs = []
        self.vs = []
        self.dones = [done]

    def add_transition(self, ob, a, r, v, done):
        self.obs.append(ob)
        self.acts.append(a)
        self.rs.append(r)
        self.vs.append(v)
        self.dones.append(done)

    def _add_R_Adv(self, R):
        Rs = []
        Advs = []
        # use post-step dones here
        for r, v, done in zip(self.rs[::-1], self.vs[::-1], self.dones[:0:-1]):
            R = r + self.gamma * R * (1.-done)
            Adv = R - v
            Rs.append(R)
            Advs.append(Adv)
        Rs.reverse()
        Advs.reverse()
        self.Rs = Rs
        self.Advs = Advs

    def sample_transition(self, R, discrete=True):
        self._add_R_Adv(R)
        obs = np.array(self.obs, dtype=np.float32)
        if discrete:
            acts = np.array(self.acts, dtype=np.int32)
        else:
            acts = np.array(self.acts, dtype=np.float32)
        Rs = np.array(self.Rs, dtype=np.float32)
        Advs = np.array(self.Advs, dtype=np.float32)
        # use pre-step dones here
        dones = np.array(self.dones[:-1], dtype=np.bool)
        self.reset(self.dones[-1])
        return obs, acts, dones, Rs, Advs

class PPOBuffer(TransBuffer):
    """PPOÃÃ£Â·Â¨ÃÂ¹ÃÃÂµÃÂ¹Ã¬Â¼Â£Â»ÂºÂ³Ã¥ÃÃ¸Â¡Â£
    ÃÃ Â±ÃOnPolicyBufferÂ£Â¬ÃÃ¶Â¼ÃÃÃÂ´Ã¦Â´Â¢Â¾ÃÂ²ÃÃÃÂ¸ÃÃÃÂµÃÂ¹Â¦ÃÃÂ¡Â£
    """
    def __init__(self, gamma, buffer_size):
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.reset()

    def reset(self, done=False):
        # doneÂ±ÃªÃÂ¾ÃÃÃÃÃÂ¿ÃÂ»Â²Â½ÃÂ°ÂµÃÃÂ´ÃÂ¬
        self.obs = []
        self.acts = []
        self.rs = []
        self.vs = []
        self.log_probs = []
        self.dones = [done]

    def add_transition(self, ob, a, r, v, log_prob, done):
        """ÃÃ­Â¼ÃÃÂ»Â¸Ã¶Â¹Ã¬Â¼Â£ÃÂªÂ»Â»Â¡Â£
        ÃÃÃÃ¶log_probÃÃÃÃÂ´Ã¦Â´Â¢ÂµÂ±ÃÂ°Â²ÃÃÃÃÃÂµÃÂ¶Â¯ÃÃ·Â¸ÃÃÃÂµÃÂ¶ÃÃÃ½ÃÂµÂ¡Â£
        """
        self.obs.append(ob)
        self.acts.append(a)
        self.rs.append(r)
        self.vs.append(v)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def _add_R_Adv(self, R):
        """Â¼ÃÃÃ£ÃÂ¿ÃÂ»Â²Â½ÂµÃÂ»ÃÂ±Â¨Â£Â¨RÂ£Â©ÂºÃÃÃÃÃÂ£Â¨AdvantageÂ£Â©Â¡Â£"""
        Rs = []
        Advs = []
        # ÃÂ¹ÃÃpost-stepÃÃªÂ³ÃÃÂ´ÃÂ¬
        for r, v, done in zip(self.rs[::-1], self.vs[::-1], self.dones[:0:-1]):
            R = r + self.gamma * R * (1.-done)
            Adv = R - v
            Rs.append(R)
            Advs.append(Adv)
        Rs.reverse()
        Advs.reverse()
        self.Rs = Rs
        self.Advs = Advs

    def sample_transition(self, R, discrete=True):
        """Â²ÃÃÃ¹Â¹Ã¬Â¼Â£ÃÃ½Â¾ÃÂ¡Â£
        Â·ÂµÂ»ÃÂ¹ÃÂ²Ã¢Â¡Â¢Â¶Â¯ÃÃ·Â¡Â¢ÃÃªÂ³ÃÂ±ÃªÃÂ¾Â¡Â¢Â»ÃÂ±Â¨Â¡Â¢ÃÃÃÃÂºÂ¯ÃÃ½ÂºÃÂ¾ÃÂ²ÃÃÃÂ¶ÃÃÃ½Â¸ÃÃÃÂ¡Â£
        """
        self._add_R_Adv(R)
        obs = np.array(self.obs, dtype=np.float32)
        if discrete:
            acts = np.array(self.acts, dtype=np.int32)
        else:
            acts = np.array(self.acts, dtype=np.float32)
        Rs = np.array(self.Rs, dtype=np.float32)
        Advs = np.array(self.Advs, dtype=np.float32)
        log_probs = np.array(self.log_probs, dtype=np.float32)
        # ÃÂ¹ÃÃpre-stepÃÃªÂ³ÃÂ±ÃªÃÂ¾
        dones = np.array(self.dones[:-1], dtype=np.bool)
        self.reset(self.dones[-1])
        return obs, acts, dones, Rs, Advs, log_probs
        
        
# å¨TransBufferç±»ä¹åæ·»å SPPOBufferç±»
class SPPOBuffer(TransBuffer):
    """Spillback-aware PPO algorithm trajectory buffer.
    Extends PPOBuffer with spillback information storage.
    """
    def __init__(self, gamma, buffer_size):
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.n_s = None  # Will be set at first add_transition
        self.n_w = None  # Will be set at first add_transition
        self.reset()

    def reset(self, done=False):
        # The done flag is for the state before each step
        self.obs = []
        self.acts = []
        self.rs = []
        self.vs = []
        self.log_probs = []
        self.spillbacks = []  # Added to store spillback information
        self.dones = [done]

    def add_transition(self, ob, a, r, v, log_prob, done, spillback=None):
        """Add a trajectory transition.
        Additional spillback parameter for storing spillback information.
        """
        self.obs.append(ob)
        self.acts.append(a)
        self.rs.append(r)
        self.vs.append(v)
        self.log_probs.append(log_prob)
        
        # Properly handle spillback information
        if spillback is not None:
            self.spillbacks.append(spillback)
        else:
            # If no spillback is provided, create a zero vector
            # Don't rely on ob length, use a fixed size
            max_neighbors = 4  # Adjust based on actual grid structure, 5x5 grid has max 4 neighbors
            self.spillbacks.append(np.zeros(max_neighbors))
        
        self.dones.append(done)

    def _add_R_Adv(self, R):
        """Calculate Returns (R) and Advantage for each step."""
        Rs = []
        Advs = []
        # Use post-step done states
        for r, v, done in zip(self.rs[::-1], self.vs[::-1], self.dones[:0:-1]):
            R = r + self.gamma * R * (1.-done)
            Adv = R - v
            Rs.append(R)
            Advs.append(Adv)
        Rs.reverse()
        Advs.reverse()
        self.Rs = Rs
        self.Advs = Advs

    def sample_transition(self, R, discrete=True):
        """Sample trajectory data.
        Returns observations, actions, done flags, returns, advantage function,
        old policy log probabilities, and spillback information.
        """
        self._add_R_Adv(R)
        obs = np.array(self.obs, dtype=np.float32)
        if discrete:
            acts = np.array(self.acts, dtype=np.int32)
        else:
            acts = np.array(self.acts, dtype=np.float32)
        Rs = np.array(self.Rs, dtype=np.float32)
        Advs = np.array(self.Advs, dtype=np.float32)
        log_probs = np.array(self.log_probs, dtype=np.float32)
        spillbacks = np.array(self.spillbacks, dtype=np.float32)
        # Use pre-step done flags
        dones = np.array(self.dones[:-1], dtype=np.bool)
        self.reset(self.dones[-1])
        return obs, acts, dones, Rs, Advs, log_probs, spillbacks

class ReplayBuffer(TransBuffer):
    def __init__(self, buffer_size, batch_size):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.cum_size = 0
        self.buffer = []

    def add_transition(self, ob, a, r, next_ob, done):
        experience = (ob, a, r, next_ob, done)
        if self.cum_size < self.buffer_size:
            self.buffer.append(experience)
        else:
            ind = int(self.cum_size % self.buffer_size)
            self.buffer[ind] = experience
        self.cum_size += 1

    def reset(self):
        self.buffer = []
        self.cum_size = 0

    def sample_transition(self):
        # Randomly sample batch_size examples
        minibatch = random.sample(self.buffer, self.batch_size)
        state_batch = np.asarray([data[0] for data in minibatch])
        action_batch = np.asarray([data[1] for data in minibatch])
        next_state_batch = np.asarray([data[3] for data in minibatch])
        reward_batch = np.asarray([data[2] for data in minibatch])
        done_batch = np.asarray([data[4] for data in minibatch])
        return state_batch, action_batch, next_state_batch, reward_batch, done_batch

    @property
    def size(self):
        return min(self.buffer_size, self.cum_size)

"""
util functions
"""
class Scheduler:
    def __init__(self, val_init, val_min=0, total_step=0, decay='linear'):
        self.val = val_init
        self.N = float(total_step)
        self.val_min = val_min
        self.decay = decay
        self.n = 0

    def get(self, n_step):
        self.n += n_step
        if self.decay == 'linear':
            return max(self.val_min, self.val * (1 - self.n / self.N))
        else:
            return self.val


if __name__ == '__main__':
    test_layers()

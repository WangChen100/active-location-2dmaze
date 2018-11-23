# encoding: utf-8
'''
@project: active-navigation-2dmaze
@file: inference.py
@version:
@author: wangchen
@contact: wangchen100@163.com
@create_time: 18-11-22 下午4:53
'''
import math
import tensorflow as tf
import tensorflow.contrib.slim as slim

A_BOUND = math.pi
MAZE_HEIGHT = 7
MAZE_WIDTH=7
NUM_ORIENTATION=4
NUM_ACTION=4
NUM_HISTORY=8
ENTROPY_BETA=0.01


class ACNet(object):

    def __init__(self, scope, global_config=None):
        with tf.variable_scope(scope):
            self.belief = tf.placeholder(tf.float32, [None, MAZE_HEIGHT, MAZE_WIDTH, NUM_ORIENTATION+1], 'belief')
            # self.ah = tf.placeholder(tf.float32, [None, NUM_HISTORY*NUM_ACTION], 'action_history')
            # self.dh = tf.placeholder(tf.float32, [None, NUM_HISTORY], 'depth_history')
            # self.th = tf.placeholder(tf.float32, [None, NUM_HISTORY], 'time_history')

            mu, sigma, val, self.a_params, self.c_params = self._network(scope)

            if scope != 'global':  # local net, calculate losses
                globalAC, self.SESS, self.OPT = global_config
                self.a_his = tf.placeholder(tf.float32, [None, NUM_ACTION], 'actions')
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')

                td = tf.subtract(self.v_target, val, name='TD_error')

                with tf.name_scope('critic_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))

                with tf.name_scope('wrap_a_out'):
                    mu, sigma = mu*A_BOUND, sigma+1e-4

                normal_dist = tf.distributions.Normal(mu, sigma)

                with tf.name_scope('actor_loss'):
                    log_prob = normal_dist.log_prob(self.a_his)
                    exp_v = log_prob * tf.stop_gradient(td)
                    entropy = normal_dist.entropy()  # encourage exploration
                    self.exp_v = exp_v + ENTROPY_BETA * entropy
                    self.a_loss = tf.reduce_mean(-self.exp_v)

                with tf.name_scope('local_grad'):
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)

                with tf.name_scope('sync'):
                    with tf.name_scope('pull'):
                        self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                        self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                    with tf.name_scope('push'):
                        self.update_a_op = self.OPT.apply_gradients(zip(self.a_grads, globalAC.a_params))
                        self.update_c_op = self.OPT.apply_gradients(zip(self.c_grads, globalAC.c_params))

                with tf.name_scope('choose_a'):  # use local params to choose action
                    self.A = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=[0, 1]), -A_BOUND, A_BOUND)

    def _network(self, scope):
        conv1 = slim.convolution2d(self.belief, 16, tf.nn.relu, 7, 3, 'same', scope=scope+'/conv_layer1')
        conv2 = slim.convolution2d(conv1, 16, tf.nn.relu, 3, 1, 'same', scope=scope + '/conv_layer2')
        fc = slim.fully_connected(slim.flatten(conv2), 256, scope=scope + '/conv_fc')
        # fc_new = tf.concat(1, [fc, self.ah, self.dh, self.th])

        w_init = tf.random_normal_initializer(0., .1)
        with tf.variable_scope('actor'):
            a_mu = tf.layers.dense(fc, NUM_ACTION, tf.nn.tanh, kernel_initializer=w_init, name='actions')
            a_sigma = tf.layers.dense(fc, NUM_ACTION, tf.nn.softplus, kernel_initializer=w_init, name='actions')
        with tf.variable_scope('critic'):
            l_c = tf.layers.dense(fc, 1, None, kernel_initializer=w_init, name='v')  # state value

        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')

        return a_mu, a_sigma, l_c, a_params, c_params

    def update_global(self, feed_dict):  # run by a local
        self.SESS.run([self.update_a_op, self.update_c_op], feed_dict)  # local grads applies to global net

    def pull_global(self):  # run by a local
        self.SESS.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s):  # run by a local
        s = tf.expand_dims(s, axis=0, name='single2batch')
        return self.SESS.run(self.A, {self.belief: s})


class Worker(object):
    def __init__(self, name,  global_config):
        self.name = name
        self.AC = ACNet(name, global_config)

    def work(self):
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []
        while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            s = self.env.reset()
            ep_r = 0
            for ep_t in range(MAX_EP_STEP):

                a = self.AC.choose_action(s)
                s_, r, done, info = self.env.step(a)
                done = True if ep_t == MAX_EP_STEP - 1 else False

                ep_r += r
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append((r + 8) / 8)  # normalize: reward=[-16, 0] ==> reward=[-1, 1]

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    if done:
                        v_s_ = 0  # terminal
                    else:
                        v_s_ = SESS.run(self.AC.v, {self.AC.s: s_[np.newaxis, :]})[0, 0]
                    buffer_v_target = []
                    for r in buffer_r[::-1]:  # reverse buffer r
                        v_s_ = r + GAMMA * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(
                        buffer_v_target)
                    feed_dict = {
                        self.AC.s: buffer_s,
                        self.AC.a_his: buffer_a,
                        self.AC.v_target: buffer_v_target,
                    }

                    self.AC.update_global(feed_dict)
                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.AC.pull_global()

                s = s_
                total_step += 1
                if done:
                    if len(GLOBAL_RUNNING_R) == 0:  # record running episode reward
                        GLOBAL_RUNNING_R.append(ep_r)
                    else:
                        GLOBAL_RUNNING_R.append(0.9 * GLOBAL_RUNNING_R[-1] + 0.1 * ep_r)
                    print(
                        self.name,
                        "Ep:", GLOBAL_EP,
                        "| Ep_r: %i" % GLOBAL_RUNNING_R[-1],
                    )
                    GLOBAL_EP += 1
                    break
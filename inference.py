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
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from maze2d import *

A_BOUND = math.pi
NUM_ORIENTATION = 4
NUM_ACTION = 3
UPDATE_GLOBAL_ITER = 3


class ACNet(object):
    """
    This class defines actor-critic model
    """
    def __init__(self, scope, args, global_config=None):
        """
        To build graph
        :param scope:
        :param global_config:
        """
        self.args = args
        with tf.variable_scope(scope):
            self.belief = tf.placeholder(dtype=tf.float32,
                                         shape=[None, NUM_ORIENTATION+1, self.args.map_size, self.args.map_size],
                                         name='belief')
            # shape: [batch,channels,height,width] ==> [batch,height,width,channels]
            self.belief = tf.transpose(self.belief, [0, 2, 3, 1])
            # self.ah = tf.placeholder(tf.float32, [None, NUM_HISTORY*NUM_ACTION], 'action_history')
            # self.dh = tf.placeholder(tf.float32, [None, NUM_HISTORY], 'depth_history')
            # self.th = tf.placeholder(tf.float32, [None, NUM_HISTORY], 'time_history')

            self.prob, self.val, self.a_params, self.c_params = self._network(scope)

            if scope != 'global':  # local net, calculate losses
                globalAC, self.OPT, self.SESS, _ = global_config
                self.a_his = tf.placeholder(tf.float32, [None, NUM_ACTION], 'actions')
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'V_target')

                td = tf.subtract(self.v_target, self.val, name='TD_error')

                with tf.name_scope('critic_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))

                with tf.name_scope('actor_loss'):
                    log_prob = tf.reduce_sum(
                        tf.log(self.prob)*tf.one_hot(self.a_his, NUM_ACTION, dtype=tf.float32),
                        axis=1, keep_dims=True, name='log_prob')
                    exp_v = log_prob * tf.stop_gradient(td)
                    entropy = -(log_prob * self.prob).sum(1)
                    self.exp_v = exp_v + self.args.beta * entropy
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

    def _network(self, scope):
        """
        To inference forward
        :param scope: global or local agent number
        :return: action_probability, state_value, actor_parameters, critic_parameters
        """
        conv1 = slim.convolution2d(self.belief, 16, tf.nn.relu, 7, 3, 'same', scope=scope+'/conv_layer1')
        conv2 = slim.convolution2d(conv1, 16, tf.nn.relu, 3, 1, 'same', scope=scope + '/conv_layer2')
        fc = slim.fully_connected(slim.flatten(conv2), 256, scope=scope + '/conv_fc')  #
        # fc_new = tf.concat(1, [fc, self.ah, self.dh, self.th])

        w_init = tf.random_normal_initializer(0., .1)
        with tf.variable_scope('actor'):
            l_a = tf.layers.dense(fc, NUM_ACTION, tf.nn.softmax, kernel_initializer=w_init, name='actions')
        with tf.variable_scope('critic'):
            l_c = tf.layers.dense(fc, 1, kernel_initializer=w_init, name='v')  # state value

        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')

        return l_a, l_c, a_params, c_params

    def update_global(self, feed_dict):  # run by a local
        self.SESS.run([self.update_a_op, self.update_c_op], feed_dict)  # local grads applies to global net

    def pull_global(self):  # run by a local
        self.SESS.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s):  # run by a local
        prob_weights = self.SESS.run(self.prob, feed_dict={self.belief: s[np.newaxis, :]})
        return np.random.choice(NUM_ACTION, size=1, replace=False, p=prob_weights.ravel())[0]  # output_dims?


class Worker(object):
    """
    This class defines local agents
    """
    def __init__(self, name, args, global_config):
        self.name = name
        self.args = args
        self.SESS, self.coord = global_config[-1:]
        self.AC = ACNet(name, args, global_config)

    def work(self):
        """
        To train local agents, update global parameters and pull into local parameters
        """
        env = Maze2D(self.args)

        local_ep = 0
        local_step = 1
        buffer_belief, buffer_a, buffer_r, buffer_depth = [], [], [], []

        # train local agent in following loop
        while not self.coord.should_stop() and local_ep < self.args.max_ep:
            belief, depth = env.reset()
            ep_r = 0
            for ep_t in range(self.args.max_step):

                a = self.AC.choose_action(belief)
                belief_, r, done, depth = env.step(a)
                done = True if ep_t == self.args.max_step - 1 else False

                ep_r += r
                buffer_belief.append(belief)
                buffer_a.append(a)
                buffer_r.append(r)
                buffer_depth.append(depth)

                if local_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    if done:
                        v_s_ = 0  # terminal
                    else:
                        v_s_ = self.SESS.run(self.AC.v_target, {self.AC.belief: belief_[np.newaxis, :]})
                    buffer_v_target = []
                    for r in buffer_r[::-1]:  # reverse buffer r
                        v_s_ = r + self.args.gamma * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()
                    # buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(
                    #     buffer_v_target)
                    feed_dict = {
                        self.AC.belief: buffer_belief,
                        self.AC.a_his: buffer_a,
                        self.AC.v_target: buffer_v_target,
                        # self.AC.: buffer_depth
                    }

                    self.AC.update_global(feed_dict)
                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.AC.pull_global()

                belief = belief_
                local_step += 1
                if done:
                    # if len(GLOBAL_RUNNING_R) == 0:  # record running episode reward
                    #     GLOBAL_RUNNING_R.append(ep_r)
                    # else:
                    #     GLOBAL_RUNNING_R.append(0.9 * GLOBAL_RUNNING_R[-1] + 0.1 * ep_r)
                    # print(
                    #     self.name,
                    #     "Ep:", GLOBAL_EP,
                    #     "| Ep_r: %i" % GLOBAL_RUNNING_R[-1],
                    # )
                    local_ep += 1
                    break

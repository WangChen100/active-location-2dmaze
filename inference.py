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
import matplotlib.pyplot as plt
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
        if scope == 'global':
            with tf.variable_scope(scope):
                self.belief_original = tf.placeholder(dtype=tf.float32,
                                                      shape=[None, NUM_ORIENTATION + 1, self.args.map_size,
                                                             self.args.map_size],
                                                      name='original_belief_data')
                self._network()
        else:
            with tf.variable_scope(scope):
                self.belief_original = tf.placeholder(dtype=tf.float32,
                                                      shape=[None, NUM_ORIENTATION+1, self.args.map_size,
                                                             self.args.map_size],
                                                      name='original_belief_data')
                # shape: [batch,channels,height,width] ==> [batch,height,width,channels]
                self.belief = tf.transpose(self.belief_original, [0, 2, 3, 1], name='transpose_belief')
                # self.ah = tf.placeholder(tf.float32, [None, NUM_HISTORY*NUM_ACTION], 'action_history')
                # self.dh = tf.placeholder(tf.float32, [None, NUM_HISTORY], 'depth_history')
                # self.th = tf.placeholder(tf.float32, [None, NUM_HISTORY], 'time_history')

                self.prob, self.val = self._network()

                self.OPT, self.SESS, _ = global_config

                self.a_his = tf.placeholder(tf.uint8, [None, 1], 'actions')
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'V_target')

                td = tf.subtract(self.v_target, self.val, name='TD_error')

                with tf.name_scope('critic_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))

                with tf.name_scope('actor_loss'):
                    log_prob = tf.log(self.prob)*tf.one_hot(self.a_his, NUM_ACTION, dtype=tf.float32)
                    self.policy_loss = tf.reduce_sum(log_prob * td)
                    self.entropy = tf.reduce_sum(log_prob * self.prob)

                self.loss = 0.5*self.c_loss-self.policy_loss - self.args.beta * self.entropy

                with tf.name_scope('local_grad'):
                    self.local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                    self.gradients = tf.gradients(self.loss, self.local_vars)
                    self.var_norms = tf.global_norm(self.local_vars)
                    # self.grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 40.0)
                    # self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    # self.c_grads = tf.gradients(self.c_loss, self.c_params)
                    # self.sha_grads = tf.gradients(self.sha_loss, self.share_params)

            with tf.name_scope('sync'):
                self.global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                with tf.name_scope('pull'):
                    self.pull_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.local_vars, self.global_vars)]
                    # self.pull_sha_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.share_params, glo.share_params)]
                    # self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, glo.a_params)]
                    # self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, glo.c_params)]
                with tf.name_scope('push'):
                    self.apply_grads = self.OPT.apply_gradients(zip(self.gradients, self.global_vars))
                    # self.update_sha_op = self.OPT.apply_gradients(zip(self.sha_grads, glo.share_params))
                    # self.update_a_op = self.OPT.apply_gradients(zip(self.a_grads, glo.a_params))
                    # self.update_c_op = self.OPT.apply_gradients(zip(self.c_grads, glo.c_params))

    def _network(self):
        """
        To inference forward
        :param scope: global or local agent number
        :return: action_probability, state_value, actor_parameters, critic_parameters
        """
        init = tf.random_normal_initializer(0., .1)
        with tf.variable_scope('features'):
            conv = slim.repeat(self.belief_original, 2, slim.conv2d, 16, [3, 3], scope='conv')
            # conv1 = tf.nn.conv2d(self.belief, [3, 3, 5, 16], strides=[1, 2, 2,1], padding='SAME', name='conv_layer1')
            # conv2 = tf.nn.conv2d(conv1, [3, 3, 16, 16], strides=[1, 2, 2, 1], padding='SAME', name='conv_layer2')
            fc = tf.contrib.layers.fully_connected(tf.layers.flatten(conv), 256, scope='fc')
            # fc_new = tf.concat(1, [fc, self.ah, self.dh, self.th])

        with tf.variable_scope('actor'):
            l_a = tf.layers.dense(fc, NUM_ACTION, activation=tf.nn.softmax,
                                  kernel_initializer=init, name='fc')
        with tf.variable_scope('critic'):
            l_c = tf.layers.dense(fc, 1, activation=None,
                                  kernel_initializer=init, name='fc')  # state value

        # share_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='features')
        # a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor')
        # c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')

        return l_a, l_c

    def update_global(self, feed_dict):  # run by a local
        self.SESS.run([self.apply_grads], feed_dict)  # local grads applies to global net

    def pull_global(self):  # run by a local
        self.SESS.run([self.pull_params_op])

    def choose_action(self, s):  # run by a local
        prob_weights = self.SESS.run(self.prob, feed_dict={self.belief_original: s[np.newaxis, :]})
        return np.random.choice(NUM_ACTION, replace=False, p=prob_weights.ravel())  # output_dims?


class Worker(object):
    """
    This class defines local agents
    """
    def __init__(self, name, args, global_config):
        self.name = name
        self.args = args
        self.SESS, self.coord = global_config[-2:]
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
                        v_s_ = self.SESS.run(self.AC.val, {self.AC.belief_original: belief_[np.newaxis, :]})
                    buffer_v_target = []
                    for r in buffer_r[::-1]:  # reverse buffer_r
                        v_s_ = r + self.args.gamma * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()
                    # buffer_belief, buffer_a, buffer_v_target = np.array(buffer_belief), np.array(buffer_a)[:, np.newaxis], np.array(buffer_v_target)
                    feed_dict = {
                        self.AC.belief_original: np.array(buffer_belief),
                        self.AC.a_his: np.vstack(buffer_a),
                        self.AC.v_target: np.vstack(buffer_v_target),
                        # self.AC.: buffer_depth
                    }

                    self.AC.update_global(feed_dict)
                    buffer_belief, buffer_a, buffer_r = [], [], []
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

    def show_loc(self, belief):
        plt.ion()

        plt.subplot(3, 2, 1)
        plt.imshow(belief[0])

        plt.subplot(3, 2, 2)
        plt.imshow(belief[1])

        plt.subplot(3, 2, 3)
        plt.imshow(belief[2])

        plt.subplot(3, 2, 4)
        plt.imshow(belief[3])

        plt.subplot(3, 1, 2)
        plt.imshow(belief[4])

        plt.pause(0.033)
        plt.show()

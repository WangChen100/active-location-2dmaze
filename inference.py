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
    def __init__(self, scope, args):
        """
        To build graph
        :param scope:
        :param args:
        """
        self.args = args
        self.sess = tf.get_default_session()
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

                self.prob, self.val = self._network()

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

            with tf.name_scope('sync'):
                self.global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                with tf.name_scope('pull'):
                    self.pull_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.local_vars, self.global_vars)]

                with tf.name_scope('push'):
                    opt = tf.train.RMSPropOptimizer(args.lr, name='RMSPropA')
                    self.apply_grads = opt.apply_gradients(zip(self.gradients, self.global_vars))

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

        return l_a, l_c

    def update_global(self, feed_dict):  # run by a local
        self.sess.run([self.apply_grads], feed_dict)  # local grads applies to global net

    def pull_global(self):  # run by a local
        self.sess.run([self.pull_params_op])

    def choose_action(self, s):  # run by a local
        prob_weights = self.sess.run(self.prob, feed_dict={self.belief_original: s[np.newaxis, :]})
        try:
            action_id = np.random.choice(NUM_ACTION, replace=False, p=prob_weights.ravel())  # output_dims?
        except ValueError:
            print("Error: prob_weights.ravel()")
            assert True, "Error!!"
        return action_id


class Worker(object):
    """
    This class defines local agents
    """
    def __init__(self, name, args, coord):
        self.name = name
        self.args = args
        self.coord = coord
        self.AC = ACNet(name, args)
        self.sess = tf.get_default_session()

    def work(self):
        """
        To train local agents, update global parameters and pull into local parameters
        """
        env = Maze2D(self.args)

        local_ep = 1
        local_step = 1
        buffer_belief, buffer_a, buffer_r, buffer_depth = [], [], [], []

        # train local agent in following loop
        while not self.coord.should_stop() and local_ep < self.args.max_ep:
            belief, depth = env.reset()
            ep_r = 0
            accuracy = 0
            print(
                self.name,
                "Ep:", local_ep,
                "| Ep_accuracy: %f" % (accuracy/local_ep),
            )
            for ep_t in range(self.args.max_step):

                a = self.AC.choose_action(belief)
                belief_, r, done, depth, loc, label = env.step(a)
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
                        v_s_ = self.sess.run(self.AC.val, {self.AC.belief_original: belief_[np.newaxis, :]})
                    buffer_v_target = []
                    for r in buffer_r[::-1]:  # reverse buffer_r
                        v_s_ = r + self.args.gamma * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

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
                    if len(loc) == 1:
                        if self.distances(loc[0], label) < 5:
                            accuracy += 1
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

    def distances(self, p1, p2):
        """
        calculate distance between p1 and p2
        :param p1:
        :param p2:
        :return:
        """
        return np.sum((p1-p2)*(p1-p2))

# encoding: utf-8
'''
@project: active-navigation-2dmaze
@file: inference.py
@version:
@author: wangchen
@contact: wangchen100@163.com
@create_time: 18-11-22 下午4:53
'''

import tensorflow as tf

N_S = 49
N_A = 2


class ACNet(object):
    def __init__(self, scope, globalAC=None):
        with tf.variable_scope(scope):
            self.s=tf.placeholder(tf.float32, [None, N_S], 'state')
            self.actions, self.v, self.a_params, self.c_params = self._network(scope)

            if scope != 'global':  # local net, calculate losses
                self.a_his = tf.placeholder(tf.float32, [None, N_A], 'action')
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')

                td = tf.subtract(self.v_target, self.v, name='TD_error')

                with tf.name_scope('critic_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))

                with tf.name_scope('wrap_a_out'):
                    mu, sigma = mu * A_BOUND[1], sigma + 1e-4  # mu=[-2, 2], sigma != 0

                normal_dist = tf.distributions.Normal(mu, sigma)

                with tf.name_scope('actor_loss'):
                    log_prob = normal_dist.log_prob(self.a_his)
                    exp_v = log_prob * tf.stop_gradient(td)
                    entropy = normal_dist.entropy()  # encourage exploration
                    self.exp_v = ENTROPY_BETA * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)

                with tf.name_scope('choose_a'):  # use local params to choose action
                    self.A = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=[0, 1]), A_BOUND[0], A_BOUND[1])

                with tf.name_scope('local_grad'):
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)

                with tf.name_scope('sync'):
                    with tf.name_scope('pull'):
                        self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                        self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                    with tf.name_scope('push'):
                        self.update_a_op = OPT_A.apply_gradients(zip(self.a_grads, globalAC.a_params))
                        self.update_c_op = OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params))

    def _network(self, scope):
        w_init = tf.random_normal_initializer(0., .1)
        with tf.variable_scope('actor'):
            l_a = tf.layers.dense(self.s, 200, tf.nn.relu6, kernel_initializer=w_init, name='la')
            actions = tf.layers.dense(l_a, N_A, tf.nn.softplus, kernel_initializer=w_init, name='actions')
        with tf.variable_scope('critic'):
            l_c = tf.layers.dense(self.s, 100, tf.nn.relu6, kernel_initializer=w_init, name='lc')
            v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')  # state value
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        return actions, v, a_params, c_params

    def update_global(self, feed_dict):  # run by a local
      SESS.run([self.update_a_op, self.update_c_op], feed_dict)  # local grads applies to global net

    def pull_global(self):  # run by a local
      SESS.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s):  # run by a local
      s = s[np.newaxis, :]
      return SESS.run(self.A, {self.s: s})
            if scope == GLOBAL_NET_SCOPE:


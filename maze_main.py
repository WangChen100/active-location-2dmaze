# encoding: utf-8
"""
@project: active-navigation-2dmaze
@file: maze_main.py
@version:
@author: wangchen
@contact: wangchen100@163.com
@create_time: 18-11-22 下午8:25
"""
import os
import shutil
import argparse
import tensorflow as tf
import multiprocessing as mp
import threading
import inference


parser = argparse.ArgumentParser(description='Active Neural Localization')

parser.add_argument('--train', type=bool, default=True,
                    help='True(default): Train; False: Test on testing data')
parser.add_argument('--log', type=str, default="log/",
                    help='path to save graph')

# Environment arguments
parser.add_argument('-l', '--max-ep', type=int, default=30, metavar='L',
                    help='maximum length of an episode (default: 30)')
parser.add_argument('-m', '--map-size', type=int, default=21,
                    help='m(default: 7): Size of maze must be an odd natural number')

# A3C and model arguments
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--max-step', type=int, default=20, metavar='NS',
                    help='number of training iterations per training thread (default: 20)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--beta', type=float, default=0.1, metavar='BT',
                    help='parameter for entropy (default: 0.1)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--hist-size', type=int, default=5,
                    help='action history size (default: 5)')


if __name__ == '__main__':

    args = parser.parse_args()
    N_WORKERS = mp.cpu_count()

    SESS = tf.Session()

    with tf.device("/cpu:0"):
        OPT = tf.train.RMSPropOptimizer(args.lr, name='RMSPropA')
        # OPT_C = tf.train.RMSPropOptimizer(args.lr, name='RMSPropC')
        glo = inference.ACNet('global', args)  # global network object of ACNet class
        COORD = tf.train.Coordinator()

        workers = []
        # Create worker
        for i in range(N_WORKERS):
            i_name = 'W_%i' % i   # worker name
            workers.append(inference.Worker(i_name, args, (glo, OPT, SESS, COORD)))

    SESS.run(tf.global_variables_initializer())

    if os.path.exists(args.log):
        shutil.rmtree(args.log)
    tf.summary.FileWriter(args.log, SESS.graph)

    worker_threads = []
    for worker in workers:
        t = threading.Thread(target=worker.work())
        t.start()
        worker_threads.append(t)
    COORD.join(worker_threads)

    # plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    # plt.xlabel('step')
    # plt.ylabel('Total moving reward')
    # plt.show()


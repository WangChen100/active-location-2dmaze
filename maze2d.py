import os
import numpy as np
from utils.maze import *
from utils.localization import *

NUM_ORIENTATION = 4


class Maze2D(object):
    """
    Include __init__(), reset() and step()
    """
    def __init__(self, args):
        self.args = args
        if self.args.train:
            train_path = './testing_mazes/m'+str(self.args.map_size)+'_n1000.npy'
            assert os.path.exists(train_path), "training mazes do not exit"
            self.test_mazes = np.load(train_path)

    def reset(self):
        """
        reinitialize maze
        :return: state, belief (belief map adding map design), and depth which agent is seeing
        """
        # Load a test maze during evaluation
        maze_in_test_data = False
        if self.args.train:
            maze = self.test_mazes[np.random.randint(1000)]
            self.orientation = int(maze[-1])
            self.position = (int(maze[-3]), int(maze[-2]))
            self.map_design = maze[:-3].reshape(self.args.map_size, self.args.map_size)
        else:
            maze_in_test_data = True

        # Generate a maze
        while maze_in_test_data:
            # Generate a map design
            self.map_design = generate_map(self.args.map_size)

            # Get random initial position and orientation of the agent
            self.position = get_random_position(self.map_design)
            self.orientation = np.random.randint(NUM_ORIENTATION)

            maze = np.concatenate((self.map_design.flatten(),
                                   np.array(self.position),
                                   np.array([self.orientation])))

            # Make sure the maze doesn't exist in the test mazes
            if not any((maze == x).all() for x in self.test_mazes):
                # Make sure map is not symmetric
                if not (self.map_design == np.rot90(self.map_design)).all() \
                        and not (self.map_design == np.rot90(np.rot90(self.map_design))).all():
                    maze_in_test_data = False

        # -------- maze, position and orientation finished -----------------

        # Pre-compute likelihoods of all observations on the map for efficiency
        self.likelihoods = get_all_likelihoods(self.map_design)  # element = 1 indicates possible location, or not

        # Get current observation and likelihood matrix
        self.curr_depth = get_depth(self.map_design, self.position, self.orientation)
        # curr_likelihood indicates all locations with same curr_depth
        curr_likelihood = self.likelihoods[int(self.curr_depth) - 1]

        # Posterior is just the likelihood as prior is uniform
        self.belief_map = curr_likelihood  # shape=[num_orientation, map_size, map_size]

        # Renormalization of the posterior
        self.belief_map /= np.sum(self.belief_map)
        self.t = 0

        # next state for the policy model
        self.belief = np.concatenate((self.belief_map, np.expand_dims(
                                     self.map_design, axis=0)), axis=0)  # shape=[num_orientation+1, map_size, map_size]
        return self.belief, int(self.curr_depth)

    def step(self, action_id):
        """
        belief map update according to action
        :param action_id: 0:turn right; 1:turn left; 2: go forward
        :return: belief, reward, done, depth
        """
        # # Get the observation before taking the action
        # curr_depth = get_depth(self.map_design, self.position, self.orientation)

        # Posterior from last step is the prior for this step
        prior = self.belief_map

        # Transform the prior according to the action taken
        prior = transition_function(prior, self.curr_depth, action_id)

        # Calculate position and orientation after taking the action
        self.position, self.orientation = get_next_state(
            self.map_design, self.position, self.orientation, action_id)

        # Get the observation and likelihood after taking the action
        self.curr_depth = get_depth(self.map_design, self.position, self.orientation)
        curr_likelihood = self.likelihoods[int(self.curr_depth) - 1]

        # Posterior = Prior * Likelihood
        self.belief_map = np.multiply(curr_likelihood, prior)

        # Renormalization of the posterior
        self.belief_map /= np.sum(self.belief_map)

        # Calculate the reward
        reward = self.belief_map.max()
        loc = self.loc_arr()

        self.t += 1
        if self.t == self.args.max_ep:
            is_terminal = True
        else:
            is_terminal = False

        # next state for the policy model
        self.belief = np.concatenate(
            (self.belief_map, np.expand_dims(
                self.map_design, axis=0)), axis=0)

        return self.belief, reward, is_terminal, int(self.curr_depth), loc, self.position

    def loc_arr(self):
        tmp = self.belief_map.sum(axis=0)
        x = tmp.argmax() // self.args.map_size
        y = tmp.argmax() % self.args.map_size
        return np.array([x, y])


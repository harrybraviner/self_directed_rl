#! /usr/bin/python3

import tensorflow as tf
import tensorflow.contrib.slim as slim

class WorldModel:

    def __init__(self, state_space_size, action_space_size, hidden_size):

        self.state_input = tf.placeholder(shape=[None, state_space_size], dtype=tf.float32)
        self.action_input = tf.placeholder(shape=[None], dtype=tf.int32)

        self.action_input_onehot = tf.one_hot(self.action_input, action_space_size, dtype=tf.float32)

        hidden_state = slim.fully_connected(tf.concat([self.state_input, self.action_input_onehot], axis=1), hidden_size, biases_initializer=None, activation_fn=tf.nn.relu)

        self.state_output_dist = slim.fully_connected(hidden_state, state_space_size, biases_initializer=None, activation_fn=tf.nn.softmax)

if __name__ == "__main__":
    """ Self-tests for crashes. """

    import numpy as np

    world_model = WorldModel(state_space_size=4, action_space_size=2, hidden_size=5)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        feed_dict = {
            world_model.state_input: np.array([[0.1, -0.4, 1.3, -0.1]]),
            world_model.action_input: np.array([1])
        }

        output = sess.run(world_model.state_output_dist, feed_dict=feed_dict)
        print(output)

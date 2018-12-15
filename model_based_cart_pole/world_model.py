#! /usr/bin/python3

import tensorflow as tf
import tensorflow.contrib.slim as slim

class WorldModel:

    def __init__(self, state_space_size, action_space_size, hidden_size):

        self.state_input = tf.placeholder(shape=[None, state_space_size], dtype=tf.float32)
        self.action_input = tf.placeholder(shape=[None], dtype=tf.int32)

        self.action_input_onehot = tf.one_hot(self.action_input, action_space_size, dtype=tf.float32)

        hidden_state = slim.fully_connected(tf.concat([self.state_input, self.action_input_onehot], axis=1), hidden_size, biases_initializer=None, activation_fn=tf.nn.relu)
        hidden_state_2 = slim.fully_connected(hidden_state, hidden_size, biases_initializer=None, activation_fn=tf.nn.relu)

        self.state_output = slim.fully_connected(hidden_state_2, state_space_size, biases_initializer=None, activation_fn=None)

        self.state_output_ground_truth = tf.placeholder(shape=[None, state_space_size], dtype=tf.float32)

        self.loss = tf.losses.mean_squared_error(self.state_output_ground_truth, self.state_output)

        self.learning_rate = tf.placeholder(shape=[], dtype=tf.float32)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_step = optimizer.minimize(self.loss)

    def train_on_episodes(self, state_input, action_input, state_output, learning_rate=1e-2, sess=None):
        feed_dict={
            self.state_input: state_input,
            self.action_input: action_input,
            self.state_output_ground_truth: state_output,
            self.learning_rate: learning_rate
        }

        # FIXME - how should I get the session in here?
        # Should this function actually return a step?
        _, training_loss = sess.run([self.train_step, self.loss], feed_dict=feed_dict)
        return training_loss


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

        output = sess.run(world_model.state_output, feed_dict=feed_dict)
        print(output)

        world_model.train_on_episodes(state_input=np.array([[0.1, -0.4, 1.3, -0.1]]), action_input=np.array([1]),
                                      state_output=np.array([[0.4, -0.2, 1.5, 0.0]]), sess=sess)

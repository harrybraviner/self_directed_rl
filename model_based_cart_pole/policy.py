#! /usr/bin/python3

import re
import tensorflow as tf
import tensorflow.contrib.slim as slim

class Policy:
    
    def __init__(self, state_space_size, action_space_size, hidden_size):
    
        with tf.name_scope(name=None, default_name="policy_net") as scope:
            self.state_input = tf.placeholder(shape=[None, state_space_size], dtype=tf.float32)
            
            W1 = tf.Variable(tf.truncated_normal(shape=[state_space_size, hidden_size], mean=0.0, stddev=0.01, dtype=tf.float32), name="weight_1")
            hidden_layer = tf.nn.relu(tf.matmul(self.state_input, W1))

            W2 = tf.Variable(tf.truncated_normal(shape=[hidden_size, action_space_size], mean=0.0, stddev=0.01, dtype=tf.float32), name="weight_2")
            self.action_output =tf.nn.softmax(tf.matmul(hidden_layer, W2))

            # Discounted rewards are inserted here when computing gradients
            self.reward_holder = tf.placeholder(shape=[None], dtype=tf.float32)
            # The actions that were actually selected by the policy are inserted here when computing gradients
            self.action_holder = tf.placeholder(shape=[None], dtype=tf.int32)

            # For each step, get the node corresponding to the selected action
            indices_of_responsible_outputs = tf.range(0, tf.shape(self.action_output)[0]) * action_space_size + self.action_holder 
            self.responsible_outputs = tf.gather(tf.reshape(self.action_output, [-1]), indices_of_responsible_outputs)

            # Define the loss
            self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs) * self.reward_holder)

            # Make a record of the trainable variable associated to the policy (there will be others in the model)
            self.trainable_variables = tf.trainable_variables(scope=scope)

            # We'll accumulate gradients for several episodes, and use these placeholders to pass them to the optimizer
            self.gradient_holders = []
            # Annoying procedure to get "_grad_holder" added onto the end of the variable names
            name_pattern = re.compile("([a-zA-Z0-9_]+)/([a-zA-Z0-9_]+):([0-9]+)")
            for idx, tvar in enumerate(self.trainable_variables):
                match = name_pattern.match(tvar.name)
                name_holder = match.group(2) + "_grad_holder"
                self.gradient_holders.append(tf.placeholder(shape=tvar.shape, name=name_holder, dtype=tf.float32))

            # Graph node to compute gradients. Will be used in each episode.
            self.gradients = tf.gradients(self.loss, self.trainable_variables)

            self.learning_rate = tf.placeholder(shape=[], dtype=tf.float32)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.update_using_gradients = optimizer.apply_gradients(zip(self.gradient_holders, self.trainable_variables))

    def run_episode_and_accumulate_gradients(self, env, max_length=999):
        raise NotImplementedError()
            
    def apply_accumulated_gradients(self, learning_rate):
        raise NotImplementedError()

if __name__ == "__main__":

    policy = Policy(4, 2, 8)

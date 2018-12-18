#! /usr/bin/python3

import re
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

class Policy:
    
    def __init__(self, sess, state_space_size, action_space_size, hidden_size):
    
        with tf.name_scope(name=None, default_name="policy_net") as scope:
            self.rng = np.random.RandomState(12345) # Needed to realise episodes
            self.sess = sess # Referenced later

            self.state_input = tf.placeholder(shape=[None, state_space_size], dtype=tf.float32)
            
            W1 = tf.Variable(tf.truncated_normal(shape=[state_space_size, hidden_size], mean=0.0, stddev=0.01, dtype=tf.float32), name="weight_1")
            hidden_layer = tf.nn.relu(tf.matmul(self.state_input, W1))

            W2 = tf.Variable(tf.truncated_normal(shape=[hidden_size, action_space_size], mean=0.0, stddev=0.01, dtype=tf.float32), name="weight_2")
            self.action_output = tf.nn.softmax(tf.matmul(hidden_layer, W2))

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

            # We'll accumulate gradients for several episodes here...
            self.grad_buffers = [np.zeros(shape=grad.shape, dtype=np.float32) for grad in self.trainable_variables]
            # ...and use these placeholders to pass them to the optimizer
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

    def apply(self, state):
        feed_dict = { self.state_input: [state] }
        action_dist = self.sess.run(self.action_output, feed_dict=feed_dict)
        chosen_action = self.rng.choice(action_dist.shape[1], p=action_dist[0])
        return chosen_action

    @staticmethod
    def discount_rewards(rewards, gamma):
        discounted_rewards = np.zeros_like(rewards, dtype=np.float32)
        running_discount = 0.0
        for idx in range(len(rewards)-1, -1, -1):
            running_discount *= gamma
            running_discount += rewards[idx]
            discounted_rewards[idx] = running_discount
        return discounted_rewards

    def run_episode_and_accumulate_gradients(self, env, reward_gamma=0.99, max_steps=1000):
        s = env.reset()
        done = False

        ep_history = []

        step = 0
        while (not done) and (max_steps is None or step < max_steps):
            step += 1

            # Run the policy to choose an action
            chosen_action = self.apply(s)

            # Step the environment
            s1, r, done, _ = env.step(chosen_action)
            # Record in the episode history
            ep_history.append([s, s1, r, chosen_action])
            s = s1

        ep_history = np.array(ep_history)

        # Compute gradients
        discounted_rewards = Policy.discount_rewards(ep_history[:, 2], reward_gamma)
        feed_dict = {
            self.state_input: np.stack(ep_history[:, 0], axis=0),
            self.reward_holder: discounted_rewards,
            self.action_holder: ep_history[:, 3],
        }
        gradients = self.sess.run(self.gradients, feed_dict=feed_dict)
        # Add these gradients to the grad buffers
        for grad_buffer, grad in zip(self.grad_buffers, gradients):
            grad_buffer += grad

        # Return some output that the user can decide how to collate
        total_reward = np.sum(ep_history[:,2])

        return total_reward

    def apply_accumulated_gradients(self, learning_rate):

        # Associate the placeholders to the accumulated gradient values
        feed_dict = dict(zip(self.gradient_holders, self.grad_buffers))
        feed_dict[self.learning_rate] = learning_rate
        # Run the update step
        self.sess.run(self.update_using_gradients, feed_dict=feed_dict)

        # Clear the gradient buffers
        for grad_buffer in self.grad_buffers:
            grad_buffer *= 0.0

if __name__ == "__main__":
    import gym
    from ewma import EWMA

    total_episodes = 5000
    episodes_per_update = 5
    learning_rate = 1e-2

    with tf.Session() as sess:
        policy = Policy(sess, 4, 2, 8)

        sess.run(tf.global_variables_initializer())

        env = gym.make("CartPole-v0")

        reward_smoothed = EWMA(0.99)

        for episode in range(total_episodes + 1):
            reward = policy.run_episode_and_accumulate_gradients(env)

            reward_smoothed.update(reward)

            if episode > 0 and episode % episodes_per_update == 0:
                policy.apply_accumulated_gradients(learning_rate)

            if episode % 100 == 0:
                print("Smoothed reward: {}".format(reward_smoothed.value()))


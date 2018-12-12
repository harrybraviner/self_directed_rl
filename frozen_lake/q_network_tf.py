import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')

# Define up the network
tf.reset_default_graph()

q_net_input = tf.placeholder(shape=[1, 16], dtype=tf.float32)
q_net_W = tf.Variable(tf.random_uniform([16, 4], 0, 0.01))
q_net_output = tf.matmul(q_net_input, q_net_W)
q_net_greedy_prediction = tf.argmax(q_net_output, 1)

q_net_next_q = tf.placeholder(shape=[1, 4], dtype=tf.float32)
loss = tf.reduce_sum(tf.square(q_net_next_q - q_net_output))
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
update_step = trainer.minimize(loss)

init = tf.initialize_all_variables()

# Hyperparameters
reward_discount = 0.99
exploration_threshold = 0.1
num_episodes = 10000

episode_length_list = []
reward_list = []

with tf.Session() as sess:
    sess.run(init)

    for episode in range(1, num_episodes + 1):
        # Reset the environment
        s = env.reset()
        total_reward = 0.0
        done = False
        step = 0

        while step < 99 and not done:
            step += 1
            chosen_action, allQ = sess.run([q_net_greedy_prediction, q_net_output],
                                           feed_dict={
                                   q_net_input: np.identity(16)[s:s+1]
                               })
            # With some probability, take a random step instead
            if np.random.rand(1) < exploration_threshold:
                chosen_action[0] = env.action_space.sample()

            # Step the environment
            s1, reward, done, _ = env.step(chosen_action[0])

            Q1 = sess.run(q_net_output,
                          feed_dict={
                              q_net_input: np.identity(16)[s1:s1+1]
                          })

            # Get the target for our supervised learning
            maxQ1 = np.max(Q1)
            targetQ = allQ # This means that we aren't even going to train the other output neurons
            targetQ[0, chosen_action[0]] = reward + reward_discount*maxQ1

            # Take a training step
            _, W1 = sess.run([update_step, q_net_W],
                             feed_dict={
                                 q_net_input: np.identity(16)[s:s+1],
                                 q_net_next_q: targetQ
                             })

            total_reward += reward
            s = s1

        if episode % 1000 == 0:
            print("Episode: " + str(episode))

        episode_length_list.append(step)
        reward_list.append(total_reward)

# Plot the smoothed performance
smoothed_running_reward = 0.0
smoothing_parameter = 0.01
smoothed_reward_list = [0.0] * len(reward_list)
for (i, r) in enumerate(reward_list):
    smoothed_running_reward *= (1.0 - smoothing_parameter)
    smoothed_running_reward += smoothing_parameter * r
    smoothed_reward_list[i] = smoothed_running_reward

plt.plot(smoothed_reward_list)
plt.ylabel('Episode reward')
plt.show()

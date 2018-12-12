import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')

learning_rate = 0.8
reward_discount = 0.95
num_episodes = 50000

Q = np.zeros([env.observation_space.n, env.action_space.n])

reward_list = []

for i in range(num_episodes):
    s = env.reset()
    reward_total = 0.0
    done = False
    step = 0
    while step < 100 and not done:
        step += 1

        # Choose an action greedily with some noise
        chosen_action = np.argmax(Q[s, :] + (1.0/(i+1))*np.random.rand(1, env.action_space.n))

        # Step the environment
        s1, reward, done, _ = env.step(chosen_action)

        # Update the Q-table
        Q[s, chosen_action] += learning_rate*(reward + reward_discount*np.max(Q[s1, :]) - Q[s, chosen_action])

        s = s1
        reward_total += reward

    if i % 5000 == 0:
        print("Episode: " + str(i))

    reward_list.append(reward_total)

smoothed_running_reward = 0.0
smoothing_parameter = 0.001
smoothed_reward_list = [0.0] * len(reward_list)
for (i, r) in enumerate(reward_list):
    smoothed_running_reward *= (1.0 - smoothing_parameter)
    smoothed_running_reward += smoothing_parameter * r
    smoothed_reward_list[i] = smoothed_running_reward

plt.plot(smoothed_reward_list)
plt.ylabel('Episode reward')
plt.show()

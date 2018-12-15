#! /usr/bin/python3

import tensorflow as tf
import numpy as np
import gym
from world_model import WorldModel

def make_episode(env, policy, max_length=None):
    states_in = []
    states_out = []
    actions = []

    s = env.reset()
    done = False

    while (not done) and (max_length is None or length < max_length):
        a = policy(s)
        s1, reward, done, _ = env.step(a)

        states_in.append(s)
        states_out.append(s1)
        actions.append(a)

        s = s1

    states_in = np.stack(states_in, axis=0)
    states_out = np.stack(states_out, axis=0)
    actions = np.stack(actions, axis=0)

    return states_in, states_out, actions

def main():
    
    env = gym.make("CartPole-v0")
    action_space_size = env.action_space.n
    state_space_size = env.observation_space.shape[0]

    world_model = WorldModel(state_space_size, action_space_size, hidden_size=32)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    rng = np.random.RandomState(123)
    policy = lambda s : rng.choice(2)

    smoothing_gamma = 0.01
    smoothed_loss = None

    for batch in range(20001):
        states_in, states_out, actions = make_episode(env, policy)

        loss = world_model.train_on_episodes(states_in, actions, states_out, learning_rate=1e-2, sess=sess)

        if smoothed_loss is None:
            smoothed_loss = loss
        else:
            smoothed_loss *= (1.0 - smoothing_gamma)
            smoothed_loss += smoothing_gamma * loss

        if batch % 500 == 0:
            print(smoothed_loss)

if __name__ == "__main__":
    main()

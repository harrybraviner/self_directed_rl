#! /usr/bin/python3

import tensorflow as tf
import numpy as np
import gym
from world_model import WorldModel
from policy import Policy


def main():

    with tf.Session() as sess:
        model_learning_rate = 1e-2
        model_hidden_size = 8
        model_training_episodes_per_batch = 5
        model_training_batches_per_training = 10

        policy_learning_rate = 1e-2
        policy_hidden_size = 8
        policy_training_episodes_per_batch = 5
        policy_training_batches_per_training = 10

        evaluation_episodes = 10
        num_rounds = 1000

        env = gym.make('CartPole-v0')
        state_space_size = env.observation_space.shape[0]
        action_space_size = env.action_space.n
    
        world_model = WorldModel(state_space_size, action_space_size, model_hidden_size)
        policy = Policy(sess, state_space_size, action_space_size, policy_hidden_size)

        sess.run(tf.global_variables_initializer())

        def make_episode_batch(env, policy, batch_size, max_length=None):
            states_in = []
            states_out = []
            actions = []

            for b in range(batch_size):
                s = env.reset()
                done = False
                length = 0
                while (not done) and (max_length is None or length < max_length):
                    length += 1
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
    

        for r in range(1, num_rounds+1):
            # Train the world model on episodes generated using the policy
            states_in, states_out, actions = make_episode_batch(env, policy.apply, model_training_episodes_per_batch)
            model_loss = world_model.train_on_episodes(states_in, actions, states_out, learning_rate=1e-2, sess=sess)
            print("Model MSE: {}".format(model_loss))

            # Train the policy on the world model
            for b in range(policy_training_batches_per_training):
                for ep in range(policy_training_episodes_per_batch):
                    policy.run_episode_and_accumulate_gradients(world_model.env_analogue)
                policy.apply_accumulated_gradients(policy_learning_rate)
    

if __name__ == "__main__":
    main()

#! /usr/bin/python3

import tensorflow as tf
import numpy as np
import gym
from world_model import WorldModel
from policy import Policy

class CircularBuffer:
    """ Used for holding initial states """

    def __init__(self, max_len):
        self.max_len = max_len
        self.memory = []
        self.insertion_idx = 0
        self.get_idx = 0

    def put(self, x):
        if len(self.memory) < self.max_len:
            self.memory.append(x)
        else:
            self.memory[self.insertion_idx] = x
            self.insertion_idx += 1
            self.insertion_idx %= self.max_len

    def get(self):
        outval = self.memory[self.get_idx]
        self.get_idx += 1
        self.get_idx %= len(self.memory)
        return outval

def main():

    with tf.Session() as sess:
        model_learning_rate = 1e-2
        model_hidden_size = 8
        model_training_episodes_per_batch = 5
        model_training_batches_per_training = 100

        policy_learning_rate = 1e-2
        policy_hidden_size = 8
        policy_training_episodes_per_batch = 5
        policy_training_batches_per_training = 10
        policy_evaluation_episodes = 20

        evaluation_episodes = 10
        num_rounds = 1000

        env = gym.make('CartPole-v0')
        state_space_size = env.observation_space.shape[0]
        action_space_size = env.action_space.n
    
        world_model = WorldModel(state_space_size, action_space_size, model_hidden_size)
        policy = Policy(sess, state_space_size, action_space_size, policy_hidden_size)

        start_state_buffer = CircularBuffer(20)
        state_initializer = lambda: start_state_buffer.get()

        sess.run(tf.global_variables_initializer())

        def make_episode_batch(env, policy, batch_size, max_length=None):
            """ Uses a black-box policy to generate an epsiode for training the model. """
            states_in = []
            states_out = []
            actions = []
            rewards = []
            dones = []

            for b in range(batch_size):
                states_in_this_ep = []
                states_out_this_ep = []
                actions_this_ep = []
                rewards_this_ep = []
                dones_this_ep = []

                s = env.reset()
                done = False
                length = 0
                while (not done) and (max_length is None or length < max_length):
                    length += 1
                    a = policy(s)
                    s1, reward, done, _ = env.step(a)

                    states_in_this_ep.append(s)
                    states_out_this_ep.append(s1)
                    actions_this_ep.append(a)
                    rewards_this_ep.append([reward])
                    dones_this_ep.append([1.0 if done else 0.0])

                    s = s1

                states_in_this_ep = np.stack(states_in_this_ep, axis=0)
                states_out_this_ep = np.stack(states_out_this_ep, axis=0)
                actions_this_ep = np.stack(actions_this_ep, axis=0)
                rewards_this_ep = np.stack(rewards_this_ep, axis=0)
                dones_this_ep = np.stack(dones_this_ep, axis=0)

                states_in.append(states_in_this_ep)
                states_out.append(states_out_this_ep)
                actions.append(actions_this_ep)
                rewards.append(rewards_this_ep)
                dones.append(dones_this_ep)

            return states_in, states_out, actions, rewards, dones
    

        for r in range(1, num_rounds+1):
            # Train the world model on episodes generated using the policy
            model_loss = [0.0, 0.0, 0.0, 0.0]
            for b in range(model_training_batches_per_training):
                states_in, states_out, actions, rewards, dones = make_episode_batch(env, policy.apply, model_training_episodes_per_batch)
                for start_state in [x[0] for x in states_in]:
                    #print(start_state)
                    start_state_buffer.put(start_state)
                this_loss = world_model.train_on_episodes(np.concatenate(states_in, axis=0),
                                                          np.concatenate(actions, axis=0),
                                                          np.concatenate(states_out, axis=0),
                                                          np.concatenate(rewards, axis=0),
                                                          np.concatenate(dones, axis=0), learning_rate=1e-2, sess=sess)
                model_loss = [x + this_loss[i] for (i, x) in enumerate(model_loss)]
            model_loss = [x / model_training_batches_per_training for x in model_loss]
            print("Model MSE: {}".format(model_loss))

            # Train the policy on the world model
            total_reward = 0.0
            for b in range(policy_training_batches_per_training):
                for ep in range(policy_training_episodes_per_batch):
                    total_reward += policy.run_episode_and_accumulate_gradients(world_model.env_analogue(sess, state_initializer=state_initializer))
                policy.apply_accumulated_gradients(policy_learning_rate)
            total_reward /= (policy_training_batches_per_training * policy_training_episodes_per_batch)
            print("Policy reward in model: {}".format(total_reward))

            # Evaluate the policy on the real environment
            evaluation_reward = 0.0
            for ep in range(policy_evaluation_episodes):
                evaluation_reward += policy.run_episode_and_accumulate_gradients(env)
            policy.clear_grad_buffers()
            evaluation_reward /= policy_evaluation_episodes
            print("Policy reward in real env: {}".format(evaluation_reward))

    

if __name__ == "__main__":
    main()

#! /usr/bin/python3

import tensorflow as tf
import tensorflow.contrib.slim as slim

class EnvAnalogue:

    def __init__(self, sess, state_stepper, state_initializer):
        self.state_stepper = state_stepper
        self.state_initializer = state_initializer
        self.sess = sess
        self.state = None

    def reset(self):
        self.state = self.state_initializer()
        return self.state

    def step(self, action):
        new_state = self.state_stepper(self.sess, self.state, action)
        self.state = new_state
        return new_state

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

    def env_analogue(self, sess, state_initializer):
        def state_stepper(sess, state, action):
            feed_dict = {self.state_input: [state], self.action_input: [action]}
            return sess.run(self.state_output, feed_dict=feed_dict)

        return EnvAnalogue(sess, state_stepper, state_initializer)

    def train_on_episodes(self, state_input, action_input, state_output, learning_rate, sess):
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
                                      state_output=np.array([[0.4, -0.2, 1.5, 0.0]]), learning_rate=1e-2, sess=sess)
    print("Passed a single training step without crashing.\n")

    # Show that we can train the model
    import gym
    tf.reset_default_graph()

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

    env = gym.make("CartPole-v0")
    action_space_size = env.action_space.n
    state_space_size = env.observation_space.shape[0]

    print("Attempting training on many episodes (with random choices of action). Loss should decrease.")

    world_model = WorldModel(state_space_size, action_space_size, hidden_size=32)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        rng = np.random.RandomState(123)
        policy = lambda s : rng.choice(2)

        smoothing_gamma = 0.01
        smoothed_loss = None

        for batch in range(3001):
            states_in, states_out, actions = make_episode(env, policy)

            loss = world_model.train_on_episodes(states_in, actions, states_out, learning_rate=1e-2, sess=sess)

            if smoothed_loss is None:
                smoothed_loss = loss
            else:
                smoothed_loss *= (1.0 - smoothing_gamma)
                smoothed_loss += smoothing_gamma * loss

            if batch % 500 == 0:
                print("Episode: {}\tSmoothed loss: {}".format(batch, smoothed_loss))

        print("Typically the final training loss should be < 0.01.")


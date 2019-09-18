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
        new_state, reward, done = self.state_stepper(self.sess, self.state, action)
        self.state = new_state[0]
        return new_state[0], reward[0][0], (done[0][0] > 0.5), {}

class WorldModel:

    def __init__(self, state_space_size, action_space_size, hidden_size, dropout_keep_prob=1.0):

        self.state_input = tf.placeholder(shape=[None, state_space_size], dtype=tf.float32)
        self.action_input = tf.placeholder(shape=[None], dtype=tf.int32)

        self.action_input_onehot = tf.one_hot(self.action_input, action_space_size, dtype=tf.float32)
        self.dropout_keep_prob = tf.placeholder(shape=[], dtype=tf.float32)
        self.training_keep_prob = dropout_keep_prob

        W1 = tf.Variable(tf.truncated_normal(shape=[self.state_input.shape[1].value + self.action_input_onehot.shape[1].value, hidden_size], dtype=tf.float32))
        b1 = tf.Variable(tf.zeros(shape=[hidden_size]))
        hidden_state_pre = tf.nn.relu(tf.matmul(tf.concat([self.state_input, self.action_input_onehot], axis=1), W1) + b1)
        #hidden_state_pre = slim.fully_connected(tf.concat([self.state_input, self.action_input_onehot], axis=1), hidden_size, biases_initializer=None, activation_fn=tf.nn.relu)
        hidden_state = tf.nn.dropout(hidden_state_pre, self.dropout_keep_prob)
        hidden_state_2_pre = slim.fully_connected(hidden_state, hidden_size, biases_initializer=tf.zeros_initializer(), activation_fn=tf.nn.relu)
        hidden_state_2 = tf.nn.dropout(hidden_state_2_pre, self.dropout_keep_prob)

        self.state_output = slim.fully_connected(hidden_state_2, state_space_size, biases_initializer=tf.zeros_initializer(), activation_fn=None)
        self.reward_output = slim.fully_connected(hidden_state_2, 1, biases_initializer=tf.zeros_initializer(), activation_fn=tf.nn.sigmoid)
        self.done_logits = slim.fully_connected(hidden_state_2, 1, biases_initializer=tf.zeros_initializer(), activation_fn=None)
        self.done_output = tf.nn.sigmoid(self.done_logits)

        self.state_output_ground_truth = tf.placeholder(shape=[None, state_space_size], dtype=tf.float32)
        self.reward_ground_truth = tf.placeholder(shape = [None, 1], dtype=tf.float32)
        self.done_ground_truth = tf.placeholder(shape = [None, 1], dtype=tf.float32)

        self.state_loss = tf.losses.mean_squared_error(self.state_output_ground_truth, self.state_output)
        self.reward_loss = tf.losses.mean_squared_error(self.reward_ground_truth, self.reward_output)
        # FIXME - should REALLY make this a classification loss
        # self.done_loss = - tf.reduce_mean(self.done_ground_truth * tf.log(self.done_output) + (1.0 - self.done_ground_truth) * tf.log(1.0 - self.done_output))
        self.done_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=self.done_ground_truth,
                                                         logits=self.done_logits)

        self.l2_loss = tf.nn.l2_loss(W1)
        self.loss = self.state_loss + 10*self.reward_loss + 10*self.done_loss + 1e-5*self.l2_loss

        self.learning_rate = tf.placeholder(shape=[], dtype=tf.float32)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_step = optimizer.minimize(self.loss)

    def env_analogue(self, sess, state_initializer):
        def state_stepper(sess, state, action):
            feed_dict = {self.state_input: [state], self.action_input: [action], self.dropout_keep_prob: 1.0}
            return sess.run([self.state_output, self.reward_output, self.done_output], feed_dict=feed_dict)

        return EnvAnalogue(sess, state_stepper, state_initializer)

    def train_on_episodes(self, state_input, action_input, state_output, reward, done, learning_rate, sess):
        feed_dict={
            self.state_input: state_input,
            self.action_input: action_input,
            self.state_output_ground_truth: state_output,
            self.reward_ground_truth: reward,
            self.done_ground_truth: done,
            self.learning_rate: learning_rate,
            self.dropout_keep_prob: self.training_keep_prob,
        }

        # FIXME - how should I get the session in here?
        # Should this function actually return a step?
        _, training_loss, state_loss, reward_loss, done_loss =\
            sess.run([self.train_step, self.loss, self.state_loss, self.reward_loss, self.done_loss], feed_dict=feed_dict)
        # print(sess.run([self.done_logits], feed_dict=feed_dict))
        return training_loss, state_loss, reward_loss, done_loss


if __name__ == "__main__":
    """ Self-tests for crashes. """

    import numpy as np

    world_model = WorldModel(state_space_size=4, action_space_size=2, hidden_size=5)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        feed_dict = {
            world_model.state_input: np.array([[0.1, -0.4, 1.3, -0.1]]),
            world_model.action_input: np.array([1]),
            world_model.dropout_keep_prob: 1.0
        }

        output = sess.run([world_model.state_output, world_model.reward_output, world_model.done_output], feed_dict=feed_dict)

        world_model.train_on_episodes(state_input=np.array([[0.1, -0.4, 1.3, -0.1]]), action_input=np.array([1]),
                                      state_output=np.array([[0.4, -0.2, 1.5, 0.0]]),
                                      reward=np.array([[0.5]]), done=np.array([[1.0]]), learning_rate=1e-2, sess=sess)
    print("Passed a single training step without crashing.\n")

    # Show that we can train the model
    import gym
    from ewma import EWMA
    tf.reset_default_graph()

    def make_episode(env, policy, max_length=None):
        states_in = []
        states_out = []
        actions = []
        rewards = []
        dones = []

        s = env.reset()
        done = False

        while (not done) and (max_length is None or length < max_length):
            a = policy(s)
            s1, reward, done, _ = env.step(a)

            states_in.append(s)
            states_out.append(s1)
            actions.append(a)
            rewards.append([reward])
            dones.append([1.0 if done else 0.0])

            s = s1

        states_in = np.stack(states_in, axis=0)
        states_out = np.stack(states_out, axis=0)
        actions = np.stack(actions, axis=0)
        rewards = np.stack(rewards, axis=0)
        dones = np.stack(dones, axis=0)

        return states_in, states_out, actions, rewards, dones

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
        smoothed_state_loss = EWMA(0.99)
        smoothed_reward_loss = EWMA(0.99)
        smoothed_done_loss = EWMA(0.99)

        for batch in range(3001):
            states_in, states_out, actions, rewards, dones = make_episode(env, policy)

            _, state_loss, reward_loss, done_loss =\
                    world_model.train_on_episodes(states_in, actions, states_out, rewards, dones, learning_rate=1e-2, sess=sess)

            smoothed_state_loss.update(state_loss)
            smoothed_reward_loss.update(reward_loss)
            smoothed_done_loss.update(done_loss)

            if batch % 500 == 0:
                print("Episode: {}\tSmoothed losses: State: {}\tReward: {}\tDone: {}".format(batch, smoothed_state_loss.value(),
                                                                                            smoothed_reward_loss.value(),
                                                                                            smoothed_done_loss.value()))

        print("Typically the final training loss should be < 0.01.")


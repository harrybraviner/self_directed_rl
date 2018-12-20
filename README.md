
# Simple RL - Q-Learning

Go read [https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0](Arthur Juliani's blog) Medium post about Q-Learning.

What is this doing?
The Q-table is learning the expected discounted reward of taking a particular action from each state.

# Policy based agents

See [https://medium.freecodecamp.org/an-introduction-to-policy-gradients-with-cartpole-and-doom-495b5ef2207f](the freeCodeCamp post) about policy gradients.

Maybe also look at [https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html](this). Looks like it may assume a little more intelligence on the part of the reader.

In Q-learning we estimated the expected future reward of taking action *a* form state *s*: *E[R(a,s)] = Q(a,s)*.
This is the discounted reward, assuming that we take the optimal step from all future states.
To explore we needed to add some ad-hoc stochasticity.

In policy gradients this value function, *Q*, is replaced by a policy *π*.
We take action *a* from state *s* with probability *P(a|s) = π(a|s;θ)*.
*θ* are the parameters to the policy.

How do we choose *θ*? In Q-learning the equivalent problem was to choose parameters such that *Q* approximated *E[R(a,s)]* well.
Here we'll define a score function *J(θ) = E[R;π(θ)]*, and seek to maximize this.
Note the two differences here:
* We're learning to pick actions to maximise reward, rather tan estimating expected reward and using a (greedy) heuristic to select an action.
* The expectation is now the expectation *under the policy we have learnt*, as opposed to the expectation based on our learnt value function.

Claims:
* *Policy gradients can learn stochastic policies* - Is this really true? Why will the policy not still learn to just take the action that has the greatest expected discounted reward?
   Actually, the perceptual aliasing point here is good. Maybe the optimum policy involves learning to do *different things* from the *same state* at different points in the *same episode*.

Exercises:
* Write an environment which has a single state, and two actions, *{ a1, a2 }*.
  The reward should be 0 except when action *a2* is taken immediately after action *a1*.
  What is the optimum policy? (I think this should be derivable analytically.)
  What happens if you try to fit a Q-table on this problem? What's the performance like? What's the final values in the Q-table?

# Model-based policy agents

For extra sample efficiency we can train a *model* of the environment, and then train the *policy* on that *model*.

I think this also helps in the circumstance where we can't actually interact with the real environment. Though it sounds like it would be prone to failure if there are actions that are rarely taken from a particular state, since we would not be able to fit that transition probability very well.

Example is at ./model_based_cart_pole/.

## To Do

* Why does the model network start returning NaN?
* Why does modest dropout (keep prob = 0.8) in the model network harm performance?

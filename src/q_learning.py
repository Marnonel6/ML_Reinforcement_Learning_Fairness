import numpy as np
import src.random


class QLearning:
    """
    QLearning reinforcement learning agent.

    Arguments:
      epsilon - (float) The probability of randomly exploring the action space
        rather than exploiting the best action.
      alpha - (float) The weighting to give current rewards in estimating Q. This 
        should range [0,1], where 0 means "don't change the Q estimate based on 
        current reward" 
      gamma - (float) This is the weight given to expected future rewards when 
        estimating Q(s,a). It should be in the range [0,1]. Here 0 means "don't
        incorporate estimates of future rewards into the reestimate of Q(s,a)"

      See page 131 of Sutton and Barto's Reinforcement Learning book for
        pseudocode and for definitions of alpha, gamma, epsilon 
        (http://incompleteideas.net/book/RLbook2020.pdf).  
    """

    def __init__(self, epsilon=0.2, alpha=0.5, gamma=0.5):
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

    def fit(self, env, steps=1000, num_bins=100):
        """
        Trains an agent using Q-Learning on an OpenAI Gym Environment.

        See page 131 of Sutton and Barto's book Reinforcement Learning for
        pseudocode (http://incompleteideas.net/book/RLbook2020.pdf).
        Initialize your parameters as all zeros. Choose actions with
        an epsilon-greedy approach Note that unlike the pseudocode, we are
        looping over a total number of steps, and not a total number of
        episodes. This allows us to ensure that all of our trials have the same
        number of steps--and thus roughly the same amount of computation time.

        See (https://www.gymlibrary.ml) for examples of how to use the OpenAI
        Gym Environment interface.

        In every step of the fit() function, you should sample
            two random numbers using functions from `src.random`.
            1.  First, use either `src.random.rand()` or `src.random.uniform()`
                to decide whether to explore or exploit.
            2. Then, use `src.random.choice` or `src.random.randint` to decide
                which action to choose. Even when exploiting, you should make a
                call to `src.random` to break (possible) ties.

        Please don't use `np.random` functions; use the ones from `src.random`!
        Please do not use `env.action_space.sample()`!

        Hints:
          - Use env.action_space.n and env.observation_space.n to get the
            number of available actions and states, respectively.
          - Remember to reset your environment at the end of each episode. To
            do this, call env.reset() whenever the value of "done" returned
            from env.step() is True.
          - In addition to resetting the environment, calling env.reset() will
            return the environment's initial state.
          - When choosing to exploit the best action rather than exploring,
            do not use np.argmax: it will deterministically break ties by
            choosing the lowest index of among the tied values. Instead,
            please *randomly choose* one of those tied-for-the-largest values.

        Arguments:
          env - (Env) An OpenAI Gym environment with discrete actions and
            observations. See the OpenAI Gym documentation for example use
            cases (https://www.gymlibrary.ml/content/api/).
          steps - (int) The number of actions to perform within the environment
            during training.

        Returns:
          state_action_values - (np.array) The values assigned by the algorithm
            to each state-action pair as a 2D numpy array. The dimensionality
            of the numpy array should be S x A, where S is the number of
            states in the environment and A is the number of possible actions.
          avg_rewards - (np.array) A 1D sequence of averaged rewards of length
            `num_bins`. Let s = int(np.ceil(steps / `num_bins`)), then
            rewards[0] contains the average reward over the first s steps,
            rewards[1] contains the average of the next s steps, etc.
        """
        # set up rewards list, Q(s, a) table
        n_actions, n_states = env.action_space.n, env.observation_space.n
        state_action_values = np.zeros((n_states, n_actions))
        avg_rewards = np.zeros([num_bins])
        all_rewards = []

        self.Q = np.zeros(n_actions)
        self.N = np.zeros(n_actions)

        # reset environment before your first action
        current_state = env.reset()

        # reset environment before your first action
        env.reset()

        s = int(np.ceil(steps/num_bins))
        n = 0
        w = 0
        sum_reward = 0

        for i in range(steps):

          if i == 0: # Initial step
            action_idx = src.random.choice(n_actions)
            obs, reward, done, info = env.step(action_idx)

          else:
            # self.epsilon
            # self.step_size = 1/self.N

            # Decision to explore or exploit
            decision = src.random.rand()

            if decision <= 0.5: # Explore
              action_idx = src.random.choice(n_actions)
              obs, reward, done, info = env.step(action_idx)

            else: # Exploit
              max_list = np.argwhere(self.Q == np.amax(self.Q))
              # print(f"\n max_list = {max_list}")
              if len(max_list)>1:
                action_idx = src.random.randint(len(max_list))
                action_idx = max_list[action_idx][0]
              else: # TODO Maybe delete
                action_idx = max_list[0][0]
              # print(f"\n action_idx = {max_list}")
              obs, reward, done, info = env.step(action_idx)

            # Append the reward
            all_rewards.append(reward)
            # Update amount of times this action has been executed
            self.N[action_idx] += 1
            # Update Q value for specific action
            self.Q[action_idx] = self.Q[action_idx] + self.alpha*(reward - self.gamma*self.Q[action_idx] - self.Q[action_idx])


            # print(f"\n n_actions = {n_actions}")
            # print(f"\n n_states = {n_states}")
            # print(f"\n self.Q = {self.Q}") 
            # print(f"\n self.N = {self.N}")
            # print(f"\n avg_rewards = {avg_rewards}")
            # print(f"\n avg_rewards.shape = {avg_rewards.shape}")
            # print(f"\n all_rewards = {all_rewards}")

            if w == s-1:
              avg_rewards[n] = sum_reward/s
              sum_reward = 0
              w = 0
              n += 1
            else:
              w += 1
              sum_reward += reward

            # print(f"\n n = {n}")

            if done == True:
              # Reset the environment after each step
              env.reset()

        # raise NotImplementedError
        state_action_values = np.tile(self.Q, (n_states, 1))
        return state_action_values, avg_rewards


    def predict(self, env, state_action_values):
        """
        Runs prediction on an OpenAI environment using the policy defined by
        the QLearning algorithm and the state action values. Predictions are
        run for exactly one episode. Note that one episode may produce a
        variable number of steps.

        Hints:
          - You should not update the state_action_values during prediction.
          - Exploration is only used in training. During prediction, you
            should only "exploit."
          - In addition to resetting the environment, calling env.reset() will
            return the environment's initial state
          - You should use a loop to predict over each step in an episode until
            it terminates by returning `done=True`.

        Arguments:
          env - (Env) An OpenAI Gym environment with discrete actions and
            observations. See the OpenAI Gym documentation for example use
            cases (https://www.gymlibrary.ml/content/api/).
          state_action_values - (np.array) The values assigned by the algorithm
            to each state-action pair as a 2D numpy array. The dimensionality
            of the numpy array should be S x A, where S is the number of
            states in the environment and A is the number of possible actions.

        Returns:
          states - (np.array) The sequence of states visited by the agent over
            the course of the episode. Does not include the starting state.
            Should be of length K, where K is the number of steps taken within
            the episode.
          actions - (np.array) The sequence of actions taken by the agent over
            the course of the episode. Should be of length K, where K is the
            number of steps taken within the episode.
          rewards - (np.array) The sequence of rewards received by the agent
            over the course  of the episode. Should be of length K, where K is
            the number of steps taken within the episode.
        """

        # setup
        n_actions, n_states = env.action_space.n, env.observation_space.n
        states, actions, rewards = [], [], []

        # reset environment before your first action
        current_state = env.reset()


          # Initialize
        self.states_list = np.array([])
        self.actions_list = np.array([])
        self.rewards_list = np.array([])

        print(f"\n state_action_values = {state_action_values}")

        # Exploit
        max_list = np.argwhere(self.Q == np.amax(self.Q))
        action_idx = max_list[0][0]
        if len(max_list)>1:
          action_idx = src.random.randint(len(max_list))
        obs, reward, done, info = env.step(action_idx)

        # Append
        self.states_list = np.append(self.states_list, action_idx)
        self.actions_list = np.append(self.actions_list, action_idx)
        self.rewards_list = np.append(self.rewards_list, reward)

        return self.states_list, self.actions_list, self.rewards_list

        # raise NotImplementedError

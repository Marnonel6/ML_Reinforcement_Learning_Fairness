import gym
import numpy as np


def test_bandit_slots():
    """
    Tests that the MultiArmedBandit implementation successfully finds the slot
    machine with the largest expected reward.
    """
    from src import MultiArmedBandit
    from src.random import rng
    rng.seed()

    env = gym.make('SlotMachines-v0', n_machines=10, mean_range=(-10, 10), std_range=(5, 10))
    env.seed(0)
    means = np.array([m.mean for m in env.machines])

    agent = MultiArmedBandit(epsilon=0.2)
    state_action_values, rewards = agent.fit(env, steps=10000, num_bins=100)

    assert state_action_values.shape == (1, 10)
    assert len(rewards) == 100
    assert np.argmax(means) == np.argmax(state_action_values)

    _, rewards = agent.fit(env, steps=1000, num_bins=42)
    assert len(rewards) == 42
    _, rewards = agent.fit(env, steps=777, num_bins=100)
    assert len(rewards) == 100

    states, actions, rewards = agent.predict(env, state_action_values)
    assert len(actions) == 1 and actions[0] == np.argmax(means)
    assert len(states) == 1
    assert len(rewards) == 1


def test_bandit_frozen_lake():
    """
    Tests the MultiArmedBandit implementation on the FrozenLake-v1 environment.
    """
    from src import MultiArmedBandit
    from src.random import rng
    rng.seed()

    # https://www.gymlibrary.dev/environments/toy_text/frozen_lake/
    env = gym.make('FrozenLake-v1')
    env.seed(0)

    agent = MultiArmedBandit(epsilon=0.2)
    state_action_values, rewards = agent.fit(env, steps=1000)

    assert state_action_values.shape == (16, 4)
    assert len(rewards) == 100, "Rewards should have 100 elements regardless of the number of steps"

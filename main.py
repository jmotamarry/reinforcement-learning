"""import random
import gym
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers.legacy import Adam

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

env = gym.make("InvertedDoublePendulum-v4")

states =  env.observation_space.shape[0]
actions = env.action_space


model = Sequential()
model.add(Flatten(input_shape=(1, states)))
model.add(Dense(24, activation="relu"))
model.add(Dense(24, activation="relu"))
model.add(Dense(2, activation="linear"))


agent = DQNAgent(
    model=model,
    memory=SequentialMemory(limit=50000, window_length=1),
    policy=BoltzmannQPolicy(),
    nb_actions=actions,
    nb_steps_warmup=10,
    target_model_update=0.01
)

agent.compile(Adam(lr=0.001), metrics=["mae"])
agent.fit(env, nb_steps=1000, visualize=False, verbose=1)

results = agent.test(env, nb_episodes=10, visualize=True)
print(np.mean(results.history["episode_reward"]))

env.close()"""

import numpy as np
import gym
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from tensorflow.keras.callbacks import TensorBoard
import datetime

class IteratedPrisonersDilemmaEnv(gym.Env):
    def __init__(self, rounds, rewards_matrix, memory_size):
        super(IteratedPrisonersDilemmaEnv, self).__init__()
        self.rounds = rounds
        self.rewards_matrix = rewards_matrix
        self.memory_size = memory_size
        self.action_space = gym.spaces.Discrete(2)  # Two actions: 0 and 1
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(memory_size * 2,), dtype=np.float32)
        self.memory = []

    def reset(self):
        self.memory = []
        return np.zeros(self.memory_size * 2)

    def step(self, action):
        opponent_action = np.random.choice([0, 1])
        self.memory.append((action, opponent_action))

        if len(self.memory) == self.memory_size:
            player_score = 0
            opponent_score = 0

            for i in range(self.memory_size):
                player_action, opponent_action = self.memory[i]
                player_score += self.rewards_matrix[player_action][opponent_action]
                opponent_score += self.rewards_matrix[opponent_action][player_action]

            return np.zeros(self.memory_size * 2), player_score, False, {}

        return np.zeros(self.memory_size * 2), 0, False, {}

# Define the environment
rounds = 5
rewards_matrix = np.array([[3, 0], [5, 1]])  # You can adjust the rewards matrix
memory_size = 2
env = IteratedPrisonersDilemmaEnv(rounds, rewards_matrix, memory_size)

# Define the model
model = Sequential()
model.add(Dense(16, input_shape=(1,) + env.observation_space.shape, activation='relu'))
model.add(Dense(env.action_space.n, activation='linear'))
model.add(Flatten())

# Define the memory
memory = SequentialMemory(limit=50000, window_length=1)

# Define the policy
policy = BoltzmannQPolicy()

# Define the DQN agent
dqn = DQNAgent(model=model, nb_actions=env.action_space.n, memory=memory, nb_steps_warmup=10, target_model_update=1e-2, policy=policy)

# Compile the model
dqn.compile(optimizer=Adam(), metrics=['mae'])

# Set up TensorBoard
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Training loop
for episode in range(100):  # You can adjust the number of episodes
    state = env.reset()
    done = False

    while not done:
        action = dqn.forward(state)
        next_state, reward, done, _ = env.step(action)

        # Modify the reward based on the opponent's action (optional)
        # reward -= 0.5 * env.memory[-1][1]  # Example: penalize based on opponent's action

        dqn.backward(reward, terminal=done)
        state = next_state

    # Add TensorBoard callback
    dqn.fit(env, nb_steps=1, visualize=False, verbose=0, callbacks=[tensorboard_callback])

# Test the trained agent
dqn.test(env, nb_episodes=5, visualize=True)


"""
episodes = 10
for episode in range(1, episodes + 1):
    state = env.reset()
    done = False
    score = 0

    while not done:
        action = random.choice([0, 1])
        n_step, reward, done, info = env.step(action)
        score += reward
        env.render()

    print(f"Episode {episode}, Score: {score}")

env.close()
"""

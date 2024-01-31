import numpy as np
import tensorflow as tf
import gym

class PPOAgent(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super(PPOAgent, self).__init__()
        self.policy = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(action_dim, activation='softmax')
        ])

    def call(self, state):
        return self.policy(state)

# Environment
class PrisonersDilemma(gym.Env):
    def __init__(self, rounds=10, reward_cooperate=2, reward_betray=0):
        super(PrisonersDilemma, self).__init__()
        self.action_space = gym.spaces.Discrete(2)  # 0 for cooperate, 1 for betray
        self.observation_space = gym.spaces.Discrete(1)  # Simple state representation
        self.rounds = rounds
        self.reward_cooperate = reward_cooperate
        self.reward_betray = reward_betray
        self.current_round = 0
        self.state = None

    def reset(self):
        self.current_round = 0
        self.state = np.zeros(1)  # Simple state representation
        return self.state

    def step(self, actions):
        self.current_round += 1

        # Compute rewards based on actions
        if actions[0] == 0 and actions[1] == 0:
            rewards = [self.reward_cooperate, self.reward_cooperate]
        elif actions[0] == 0 and actions[1] == 1:
            rewards = [0, 3]  # Cooperate, betray
        elif actions[0] == 1 and actions[1] == 0:
            rewards = [3, 0]  # Betray, cooperate
        else:
            rewards = [1, 1]  # Both betray

        # Update state (if needed)
        self.state = np.zeros(1)  # Update state if needed

        # Check if the maximum number of rounds is reached
        done = self.current_round >= self.rounds

        return self.state, rewards, done, {}

# Proximal Policy Optimization
def proximal_policy_optimization(env, num_episodes=1000, epochs=10, batch_size=64, gamma=0.99, epsilon=0.2):
    state_dim = env.observation_space.n
    action_dim = env.action_space.n

    agent1 = PPOAgent(state_dim, action_dim)
    agent2 = PPOAgent(state_dim, action_dim)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    for episode in range(num_episodes):
        state = env.reset()
        episode_states, episode_actions, episode_rewards = [], [], []

        for _ in range(env.rounds):
            action1_prob = agent1(state.reshape(1, -1))
            action2_prob = agent2(state.reshape(1, -1))

            action1 = np.random.choice(action_dim, p=action1_prob.numpy().flatten())
            action2 = np.random.choice(action_dim, p=action2_prob.numpy().flatten())

            next_state, rewards, done, _ = env.step([action1, action2])

            episode_states.append(state)
            episode_actions.append([action1, action2])
            episode_rewards.append(rewards)

            state = next_state

            if done:
                break

        advantages, returns = calculate_advantages_and_returns(episode_rewards, gamma)

        for epoch in range(epochs):
            indices = np.arange(len(episode_states))
            np.random.shuffle(indices)

            for start in range(0, len(episode_states), batch_size):
                batch_indices = indices[start:start + batch_size]
                batch_states = np.vstack([episode_states[i] for i in batch_indices])
                batch_actions = np.vstack([episode_actions[i] for i in batch_indices])
                batch_advantages = np.vstack([advantages[i] for i in batch_indices])
                batch_returns = np.vstack([returns[i] for i in batch_indices])

                update_ppo(agent1, optimizer, batch_states, batch_actions, batch_advantages, batch_returns, epsilon)
                update_ppo(agent2, optimizer, batch_states, batch_actions, batch_advantages, batch_returns, epsilon)

    return agent1, agent2

def calculate_advantages_and_returns(rewards, gamma):
    advantages = []
    returns = []
    adv = 0
    ret = 0

    for episode_rewards in reversed(rewards):
        delta = episode_rewards[0] - episode_rewards[1]
        adv = delta + gamma * adv
        ret = episode_rewards[0] + gamma * ret
        advantages.append([adv, -adv])  # Advantage for agent 1 and agent 2
        returns.append([ret, ret])  # Return for agent 1 and agent 2

    advantages.reverse()
    returns.reverse()

    return np.vstack(advantages), np.vstack(returns)

def update_ppo(agent, optimizer, states, actions, advantages, returns, epsilon):
    with tf.GradientTape() as tape:
        action_prob = agent(states, training=True)
        one_hot_actions = tf.one_hot(actions, depth=2)[:, :, None]  # Add an extra dimension
        chosen_action_prob = tf.reduce_sum(action_prob * one_hot_actions, axis=-1)

        ratio = tf.exp(tf.math.log(chosen_action_prob) - tf.math.log(tf.stop_gradient(chosen_action_prob)))
        clipped_ratio = tf.clip_by_value(ratio, 1 - epsilon, 1 + epsilon)

        advantages = tf.cast(advantages, dtype=tf.float32)  # Cast advantages to float32

        surrogate = tf.minimum(ratio * advantages, clipped_ratio * advantages)
        actor_loss = -tf.reduce_mean(surrogate)

        value = tf.reduce_sum(agent(states, training=True) * one_hot_actions, axis=-1)
        critic_loss = tf.reduce_mean(tf.square(value - returns))

        total_loss = actor_loss + 0.5 * critic_loss

    gradients = tape.gradient(total_loss, agent.trainable_variables)
    optimizer.apply_gradients(zip(gradients, agent.trainable_variables))

# Training
env = PrisonersDilemma(rounds=10, reward_cooperate=2, reward_betray=0)
agent1, agent2 = proximal_policy_optimization(env, num_episodes=1000)

# Testing
state = env.reset()
total_rewards = [0, 0]

for _ in range(env.rounds):
    action1_prob = agent1(state.reshape(1, -1))
    action2_prob = agent2(state.reshape(1, -1))

    action1 = np.argmax(action1_prob.numpy().flatten())
    action2 = np.argmax(action2_prob.numpy().flatten())

    next_state, rewards, done, _ = env.step([action1, action2])

    total_rewards[0] += rewards[0]
    total_rewards[1] += rewards[1]

    state = next_state

print(f"Total rewards for agent 1: {total_rewards[0]}")
print(f"Total rewards for agent 2: {total_rewards[1]}")

import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

class PolicyNetwork(tf.keras.Model):
    def __init__(self, num_actions, state_size=20):
        super(PolicyNetwork, self).__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.logits = tf.keras.layers.Dense(num_actions, activation='softmax')

    def call(self, state):
        x = self.dense1(tf.reshape(state, (1, -1)))  # Reshape to ensure at least 2 dimensions
        x = self.dense2(x)
        return self.logits(x)



class PPOAgent:
    def __init__(self, num_actions, state_size=20, memory_size=10):
        self.policy_network = PolicyNetwork(num_actions, state_size=state_size)  # Pass state_size to PolicyNetwork
        self.optimizer = tf.optimizers.Adam(learning_rate=1e-3)
        self.memory_size = memory_size
        self.state_memory = np.zeros((memory_size, state_size))

    def select_action(self, state):
        state = np.reshape(state, [1, -1]).astype(np.float32)
        probabilities = self.policy_network(state).numpy().flatten()
        action = np.random.choice(len(probabilities), p=probabilities)
        # print(f"Selected Action: {action}")
        return action


    def compute_loss(self, state, action, advantage, old_probabilities):
        state = tf.convert_to_tensor(state, dtype=tf.float32)

        probabilities = self.policy_network(state)
        action_one_hot = tf.one_hot(action, depth=probabilities.shape[-1])

        new_probabilities = self.policy_network(state)

        # Reshape action_one_hot to match the shape of new_probabilities
        action_one_hot = tf.reshape(action_one_hot, shape=new_probabilities.shape)

        ratio = tf.reduce_sum(action_one_hot * new_probabilities, axis=-1) / (tf.reduce_sum(action_one_hot * old_probabilities, axis=-1) + 1e-10)

        clipped_ratio = tf.clip_by_value(ratio, 1 - 0.2, 1 + 0.2)
        surrogate_loss = -tf.minimum(ratio * advantage, clipped_ratio * advantage)

        entropy_loss = -tf.reduce_sum(new_probabilities * tf.math.log(new_probabilities + 1e-10), axis=-1)

        return surrogate_loss + 0.01 * entropy_loss

    def train_step(self, state, action, advantage, old_probabilities):
        state = tf.convert_to_tensor(state, dtype=tf.float32)

        with tf.GradientTape() as tape:
            loss = self.compute_loss(state, action, advantage, old_probabilities)
        gradients = tape.gradient(loss, self.policy_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.policy_network.trainable_variables))

        # print(f"Probabilities: {self.policy_network(state).numpy().flatten()}, Advantage: {advantage}")




def iterated_prisoners_dilemma(agent1, agent2, num_rounds, rewards_matrix, memory_size=10):
    agent1.state_memory = np.zeros((memory_size, 2))
    agent2.state_memory = np.zeros((memory_size, 2))

    total_score_agent1 = 0
    total_score_agent2 = 0

    for _ in range(num_rounds):
        state_agent1 = agent1.state_memory.copy()
        state_agent2 = agent2.state_memory.copy()

        action_agent1 = agent1.select_action(state_agent1[-1])  # Ensure to pass the last state
        action_agent2 = agent2.select_action(state_agent2[-1])  # Ensure to pass the last state

        reward_agent1 = rewards_matrix[action_agent1, action_agent2]
        reward_agent2 = rewards_matrix[action_agent2, action_agent1]

        agent1.train_step(state_agent1[-2], action_agent1, reward_agent1, agent1.policy_network(state_agent1[-2]))
        agent2.train_step(state_agent2[-2], action_agent2, reward_agent2, agent2.policy_network(state_agent2[-2]))

        # Update state memory with the last x moves
        state_agent1 = np.concatenate([state_agent1[1:], [[action_agent1, action_agent2]]])
        state_agent2 = np.concatenate([state_agent2[1:], [[action_agent2, action_agent1]]])

        # Update total scores
        total_score_agent1 += reward_agent1
        total_score_agent2 += reward_agent2

    return total_score_agent1, total_score_agent2



def main():
    num_actions = 2
    rewards_matrix = np.array([[1, 3], [0, 2]])

    agent1 = PPOAgent(num_actions, state_size=2)  # Correct state_size parameter
    agent2 = PPOAgent(num_actions, state_size=2)  # Correct state_size parameter

    num_training_rounds = 1000
    num_rounds_per_game = 5
    scores_agent1 = []
    scores_agent2 = []

    for episode in range(num_training_rounds):
        total_score_agent1, total_score_agent2 = iterated_prisoners_dilemma(
            agent1, agent2, num_rounds=num_rounds_per_game, rewards_matrix=rewards_matrix, memory_size=10
        )

        if episode % 6 == 0:
            scores_agent1.append(total_score_agent1)
            scores_agent2.append(total_score_agent2)

        print(f"Episode: {episode + 1}, Agent 1 Score: {total_score_agent1}, Agent 2 Score: {total_score_agent2}")

        # Display moves and total scores for the last game
        # print(f"Agent 1 Moves: {agent1.state_memory}")
        # print(f"Agent 2 Moves: {agent2.state_memory}")
        # print(f"Total Score - Agent 1: {total_score_agent1}, Agent 2: {total_score_agent2}")

    # Plotting
    plt.plot(scores_agent1, label='Agent 1')
    plt.plot(scores_agent2, label='Agent 2')
    plt.xlabel('Episode')
    plt.ylabel('Total Score')
    plt.legend()
    plt.title('Scores of Agents Over Training Episodes')
    plt.show()

if __name__ == "__main__":
    main()
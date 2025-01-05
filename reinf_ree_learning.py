import gym
import torch
import numpy as np
from torch import nn
from torch.optim import Adam

# Assuming you have the DQNetwork and ReplayMemory classes from earlier
class DQNAgent:
    def __init__(self, state_size, action_size, memory_size=10000, batch_size=64, gamma=0.99, epsilon_decay=0.995, lr=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayMemory(memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = 1.0  # Initial exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = epsilon_decay
        self.q_network = DQNetwork(state_size, action_size)
        self.target_network = DQNetwork(state_size, action_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)  # Random action (explore)
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state)
        return np.argmax(q_values.detach().numpy())  # Exploit (choose best action)

    def train(self):
        if len(self.memory) < self.batch_size:
            return  # Wait until memory has enough samples

        # Sample a batch from replay memory
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to PyTorch tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # Get current Q-values
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Calculate target Q-values using target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        # Compute loss and update the model
        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decrease epsilon to shift from exploration to exploitation
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
# Initialize the environment and agent
env = gym.make('CartPole-v1')  # Replace with your custom environment if needed
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)

# Train the agent
episodes = 1000

for episode in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        agent.train()
        state = next_state
        total_reward += reward

    agent.update_target_network()  # Update target network periodically

    print(f"Episode {episode+1}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.3f}")

torch.save(agent.q_network.state_dict(), 'dqn_model.pth')
# Load trained model for inference
model = DQNetwork(state_size, action_size)
model.load_state_dict(torch.load('dqn_model.pth'))
model.eval()

def run_inference(env, model, episodes=5):
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            state = torch.FloatTensor(state).unsqueeze(0)
            q_values = model(state)
            action = np.argmax(q_values.detach().numpy())  # Choose the best action
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward

            env.render()  # Render the environment (optional)

        print(f"Inference Episode {episode+1}/{episodes}, Total Reward: {total_reward}")

# Run inference
run_inference(env, model)

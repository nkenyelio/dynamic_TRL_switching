import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import quantization
import random
import numpy as np
from collections import namedtuple, deque
import gym

class QuantizedDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QuantizedDQN, self).__init__()
        
        # Standard layers
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
        
        # Quantization steps
        self.quant = quantization.QuantStub()  # Quantize inputs
        self.dequant = quantization.DeQuantStub()  # Dequantize outputs
        
    def forward(self, x):
        x = self.quant(x)  # Quantization
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.dequant(x)  # Dequantization
        return x

def quantize_model(model):
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')  # For inference optimization
    torch.quantization.prepare(model, inplace=True)
    torch.quantization.convert(model, inplace=True)
    return model


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class PrioritizedReplayMemory:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.memory = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0

    def push(self, *args):
        max_priority = self.priorities.max() if self.memory else 1.0
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.memory) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]
        
        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        samples = [self.memory[idx] for idx in indices]
        total = len(self.memory)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)
        return samples, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, priority in zip(batch_indices, batch_priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.memory)
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.policy_net = QuantizedDQN(state_dim, action_dim)
        self.target_net = QuantizedDQN(state_dim, action_dim)
        self.memory = PrioritizedReplayMemory(10000)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.gamma = 0.99
        self.batch_size = 64
        self.update_target_every = 10
        self.steps_done = 0

        # Quantize the model
        self.policy_net = quantize_model(self.policy_net)
        self.target_net = quantize_model(self.target_net)

    def select_action(self, state, epsilon):
        if random.random() > epsilon:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                #obs, _ = env.reset()
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.action_dim)]], dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

        transitions, indices, weights = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.tensor(np.array(batch.state), dtype=torch.float32)
        action_batch = torch.tensor(batch.action)
        reward_batch = torch.tensor(batch.reward)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)))
        non_final_next_states = torch.tensor([s for s in batch.next_state if s is not None], dtype=torch.float32)

        current_q_values = self.policy_net(state_batch).gather(1, action_batch)
        next_q_values = torch.zeros(self.batch_size)
        next_q_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        expected_q_values = reward_batch + (self.gamma * next_q_values)
        loss = F.smooth_l1_loss(current_q_values, expected_q_values.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update priorities in replay memory
        priorities = np.abs(current_q_values - expected_q_values).detach().numpy() + 1e-5
        self.memory.update_priorities(indices, priorities)

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

# Example usage with OpenAI Gym:

env = gym.make('CartPole-v1')
agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)

for episode in range(100):
    state = env.reset()
    done = False
    while not done:
        action = agent.select_action(state, epsilon=0.1)
        #next_state, reward, done, _, _ = env.step(action.item())
        next_state, reward, done, _, _ = env.step(action)
        agent.memory.push(state, action, next_state if not done else None, reward)
        state = next_state

        agent.optimize_model()

    if episode % agent.update_target_every == 0:
        agent.update_target_network()

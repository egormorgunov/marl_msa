import random
import os
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from MicroserviceEnvironment import RoadEnvironment
from queries import clear_table_info, insert_agent_info

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, buffer_size=2000):
        self.memory = deque(maxlen=buffer_size)

    def store(self, experience):
        self.memory.append(experience)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# Neural Network for IQN
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 36)
        self.fc2 = nn.Linear(36, 36)
        self.fc3 = nn.Linear(36, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # Outputs Q-values for each action


# IQN Agent class
class IQNAgent:
    def __init__(self, state_size, action_size, gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate

        self.memory = ReplayBuffer()
        self.model = QNetwork(state_size, action_size).to(device)
        self.target_model = QNetwork(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.store((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).to(device)
        with torch.no_grad():
            action_values = self.model(state)
        return torch.argmax(action_values).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = self.memory.sample(batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).to(device)
            next_state = torch.FloatTensor(next_state).to(device)
            reward = torch.FloatTensor([reward]).to(device)
            done = torch.FloatTensor([done]).to(device)
            action = torch.LongTensor([action]).to(device)
            # print(f"Action1: {action1}, Model output shape: {self.model(state1).shape}")
            # print(f"Action2: {action2}, Model output shape: {self.model(state2).shape}")

            with torch.no_grad():
                target = reward
                if not done:
                    target = reward + self.gamma * torch.max(self.target_model(next_state)).unsqueeze(0)

            # Make sure current_q and target are scalars
            current_q = self.model(state)[0][action].squeeze()
            target = target.squeeze()

            # print(f"current_q: {current_q}, target: {target}")
            # print(f"current_q shape: {current_q.shape}, target shape: {target.shape}")

            loss = nn.MSELoss()(current_q, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, name):
        torch.save(self.model.state_dict(), name)


# Training parameters
num_episodes = 200
batch_size = 32

# Environment parameters
state_size = 3
action_size = 3

# Initialize environment and agents
env1 = RoadEnvironment(agent_id=1)
env2 = RoadEnvironment(agent_id=2)
agent1 = IQNAgent(state_size, action_size)
agent2 = IQNAgent(state_size, action_size)

rewards_agent1 = []
rewards_agent2 = []
counter1 = []
counter2 = []
action11 = []
action21 = []
action31 = []
action41 = []
action12 = []
action22 = []
action32 = []
action42 = []

for episode in range(num_episodes):
    print(f"Episode {episode + 1}")
    state1, state2 = env1.reset(), env2.reset()
    state1 = np.reshape(state1, [1, state_size])
    state2 = np.reshape(state2, [1, state_size])
    total_reward_agent1 = 0
    total_reward_agent2 = 0
    action_counter1 = 0
    action_counter2 = 0
    done1, done2 = False, False

    while not (done1 and done2):
        action1 = agent1.act(state1)
        action2 = agent2.act(state2)

        if not done1:
            action_counter1 += 1
        if not done2:
            action_counter2 += 1

        reward1, next_state1, done1, action1_counter1, action2_counter1, action3_counter1, action4_counter1 = env1.step(action1)
        reward2, next_state2, done2, action1_counter2, action2_counter2, action3_counter2, action4_counter2 = env2.step(action2)
        next_state1 = np.reshape(next_state1, [1, state_size])
        next_state2 = np.reshape(next_state2, [1, state_size])

        # Convert numpy types to native Python types
        reward1 = int(reward1)
        next_state1_0 = float(next_state1[0][0])
        next_state1_1 = float(next_state1[0][1])

        reward2 = int(reward2)
        next_state2_0 = float(next_state2[0][0])
        next_state2_1 = float(next_state2[0][1])

        if next_state1_0 == 7:
            reward1 -= round(action_counter1/7, 2)

        if next_state2_0 == 7:
            reward2 -= round(action_counter2/7, 2)

        # Insert the information into the database
        insert_agent_info(reward1, next_state1_0, next_state1_1, episode + 1, "public.agent1_info")
        insert_agent_info(reward2, next_state2_0, next_state2_1, episode + 1, "public.agent2_info")

        agent1.remember(state1, action1, reward1, next_state1, done1)
        agent2.remember(state2, action2, reward2, next_state2, done2)
        state1 = next_state1
        state2 = next_state2
        total_reward_agent1 += reward1
        total_reward_agent2 += reward2

        if len(agent1.memory) > batch_size:
            agent1.replay(batch_size)
            agent2.replay(batch_size)

    print(f"Episode: {episode + 1}, Agent 1 Total Reward: {total_reward_agent1}, "
          f"Agent 2 Total Reward: {total_reward_agent2}")
    rewards_agent1.append(total_reward_agent1)
    rewards_agent2.append(total_reward_agent2)
    counter1.append(action_counter1)
    counter2.append(action_counter2)
    action11.append(action1_counter1)
    action21.append(action2_counter1)
    action31.append(action3_counter1)
    action41.append(action4_counter1)
    action12.append(action1_counter2)
    action22.append(action2_counter2)
    action32.append(action3_counter2)
    action42.append(action4_counter2)

    # Update target networks at the end of each episode
    agent1.update_target_model()
    agent2.update_target_model()

print("НАГРАДА АГЕНТА 1", rewards_agent1)
print("НАГРАДА АГЕНТА 2", rewards_agent2)
print("СКОРОСТЬ АГЕНТА 1", counter1)
print("СКОРОСТЬ АГЕНТА 2", counter2)
print("НАДЕЖНОСТЬ АГЕНТА 1", action41)
print("НАДЕЖНОСТЬ АГЕНТА 2", action42)

# Save models
agent1.save("IQN_agent1.pth")
agent2.save("IQN_agent2.pth")
clear_table_info("public.agent_info")
clear_table_info("public.weather_info")

# Plotting rewards for both agents
plt.plot(rewards_agent1, label="Agent 1")
plt.plot(rewards_agent2, label="Agent 2")
plt.title("Rewards per Episode")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.legend()
plt.show()

plt.plot(counter1, label="Agent 1")
plt.plot(counter2, label="Agent 2")
plt.title("Actions per Episode")
plt.xlabel("Episode")
plt.ylabel("Number of actions")
plt.legend()
plt.show()

line1, = plt.plot([i for i in range(len(action11))], action11, label='Action = 0')
line2, = plt.plot([i for i in range(len(action21))], action21, label='Action = 1')
line3, = plt.plot([i for i in range(len(action31))], action31, label='Action = 2')
line4, = plt.plot([i for i in range(len(action41))], action41, label='Booking activated')
plt.legend()
plt.title("The chosen path for Agent 1")
plt.xlabel("Number of Episodes")
plt.ylabel("Number of actions for each pass")
plt.show()

line5, = plt.plot([i for i in range(len(action12))], action12, label='Action = 0')
line6, = plt.plot([i for i in range(len(action22))], action22, label='Action = 1')
line7, = plt.plot([i for i in range(len(action32))], action32, label='Action = 2')
line8, = plt.plot([i for i in range(len(action42))], action42, label='Booking activated')
plt.legend()
plt.title("The chosen path for Agent 2")
plt.xlabel("Number of Episodes")
plt.ylabel("Number of actions for each pass")
plt.show()
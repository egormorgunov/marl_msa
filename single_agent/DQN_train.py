import random
from collections import deque

import matplotlib.pyplot as plt
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from MicroserviceEnvironment import RoadEnvironment
from queries import (
    get_table_info,
    insert_weather_info,
    clear_table_info,
    insert_agent_info,
)

graph = []
graph1 = []
graph2 = []
graph3 = []
graph4 = []
graph5 = []

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)  # Replay memory
        self.gamma = 0.95  # Discount factor #0.95
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001  # 0.01
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(
                    self.model.predict(next_state)[0]
                )
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, name):
        self.model.save_weights(name)


# Training parameters
num_episodes = 200
batch_size = 32

# Environment parameters
state_size = 3
action_size = 3

# Initialize environment and agent
env = RoadEnvironment()
agent = DQNAgent(state_size, action_size)

for episode in range(num_episodes):
    print(f"Episode {episode + 1}")
    clear_table_info("public.booking_info")
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    total_reward = 0
    action_counter = 0
    done = False
    while not done:
        action = agent.act(state)
        action_counter += 1
        reward, next_state, done, action1_counter, action2_counter, action3_counter, action4_counter = env.step(action)
        print(reward, next_state)
        if next_state[0] == 7:
            reward -= round(action_counter/7, 2)
        insert_agent_info(reward, next_state[0], next_state[1], episode+1, "public.agent_info")
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
    print("Episode: {}, Total Reward: {}".format(episode + 1, total_reward), " \n")
    graph.append(total_reward)
    graph1.append(action_counter)
    graph2.append(action1_counter)
    graph3.append(action2_counter)
    graph4.append(action3_counter)
    graph5.append(action4_counter)

agent.save("DQN.weights.h5")
clear_table_info("public.agent_info")
clear_table_info("public.weather_info")

print(graph)
print(graph1)
print(graph2)
print(graph3)
print(graph4)
print(graph5)

plt.plot([i for i in range(len(graph))], graph)
plt.title("Training Results of DQN algorithm")
plt.xlabel("Number of Episodes")
plt.ylabel("Average Reward per Episode")
plt.show()

plt.plot([i for i in range(len(graph1))], graph1)
plt.title("Quality of Service Results")
plt.xlabel("Number of Episodes")
plt.ylabel("Number of actions per episode")
plt.show()

line1, = plt.plot([i for i in range(len(graph2))], graph2, label='Action = 0')
line2, = plt.plot([i for i in range(len(graph3))], graph3, label='Action = 1')
line3, = plt.plot([i for i in range(len(graph4))], graph4, label='Action = 2')
line4, = plt.plot([i for i in range(len(graph5))], graph5, label='Booking activated')
plt.legend()
plt.title("The chosen path")
plt.xlabel("Number of Episodes")
plt.ylabel("Number of actions for each pass")
plt.show()

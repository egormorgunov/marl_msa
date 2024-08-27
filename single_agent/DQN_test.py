import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque
from MicroserviceEnvironment import RoadEnvironment
from queries import (
    clear_weather_info,
    insert_agent_info,
    clear_agent_info,
    clear_booking_info,
)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)  # Replay memory
        self.gamma = 0.95  # Discount factor
        self.epsilon = 0.0  # No exploration during testing
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def act(self, state):
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def load(self, name):
        self.model.load_weights(name)

# Environment parameters
state_size = 3
action_size = 3

# Initialize environment and agent
env = RoadEnvironment()
agent = DQNAgent(state_size, action_size)

# Load the trained model weights
agent.load("tests/DQN.weights.h5")

num_episodes = 100  # Number of testing episodes
graph = []
graph1 = []
graph2 = []
graph3 = []
graph4 = []
graph5 = []

for episode in range(num_episodes):
    print(f"Test Episode {episode + 1}")
    clear_booking_info()
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
        insert_agent_info(reward, next_state[0], next_state[1], episode + 1)
        next_state = np.reshape(next_state, [1, state_size])
        state = next_state
        total_reward += reward
    print("Test Episode: {}, Total Reward: {}".format(episode + 1, total_reward), " \n")
    graph.append(total_reward)
    graph1.append(action_counter)
    graph2.append(action1_counter)
    graph3.append(action2_counter)
    graph4.append(action3_counter)
    graph5.append(action4_counter)

print(graph)
clear_agent_info()
clear_weather_info()

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

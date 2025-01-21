import random
import numpy as np
import tensorflow as tf
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def build_model(input_shape, output_shape):
    model = Sequential([
        Dense(24, input_dim=input_shape, activation='relu'),
        Dense(24, activation='relu'),
        Dense(output_shape, activation='linear')
    ])
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=0.001))
    return model

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # تخفیف آینده
        self.epsilon = 1.0  # میزان اکتشاف
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = build_model(state_size, action_size)
    
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
                target = (reward + self.gamma * 
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
import traci
import sumolib

def run_simulation():
    sumoCmd = ["sumo", "-c", "sumo_config.sumocfg"]
    traci.start(sumoCmd)
    step = 0
    while step < 1000:
        traci.simulationStep()
        # اینجا کد مدیریت ترافیک و کنترل چراغ‌ها را اضافه کنید
        step += 1
    traci.close()
import flwr as fl

def start_server():
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config={"num_rounds": 3},
    )

if __name__ == "__main__":
    start_server()
import flwr as fl

class TrafficLightClient(fl.client.Client):
    def __init__(self, traffic_light):
        self.traffic_light = traffic_light
    
    def get_parameters(self):
        return self.traffic_light.model.get_weights()
    
    def fit(self, parameters, config):
        self.traffic_light.model.set_weights(parameters)
        # اجرای حلقه آموزش
        train_agent(self.traffic_light)
        return self.traffic_light.model.get_weights(), len(self.traffic_light.memory), {}

    def evaluate(self, parameters, config):
        self.traffic_light.model.set_weights(parameters)
        # ارزیابی مدل
        loss = evaluate_model(self.traffic_light)
        return loss, len(self.traffic_light.memory), {}

def start_client(traffic_light):
    client = TrafficLightClient(traffic_light)
    fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=client)

if __name__ == "__main__":
    light1 = TrafficLight(1)
    start_client(light1)
def train_agent(agent, episodes=1000, batch_size=32):
    for e in range(episodes):
        state = reset_environment()  # راه‌اندازی محیط
        state = np.reshape(state, [1, agent.state_size])
        for time in range(500):
            action = agent.act(state)
            next_state, reward, done, _ = step_environment(action)  # اقدام به حرکت
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, agent.state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print(f"Episode: {e}/{episodes}, Score: {time}, Epsilon: {agent.epsilon:.2f}")
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

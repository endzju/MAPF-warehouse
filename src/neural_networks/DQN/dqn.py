import random
import time
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from src.core.MultiRobotGridEnv import MultiRobotGridEnv


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class DQNet(nn.Module):
    def __init__(self, view_shape, goal_vec_size, n_actions):
        super(DQNet, self).__init__()
        # 1. Przetwarzanie widoku (CNN) - wejście: (4, 5, 5)
        self.cnn = nn.Sequential(
            nn.Conv2d(view_shape[0], 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Obliczamy rozmiar wyjścia z CNN (dla 5x5 i padding=1 to 64 * 5 * 5)
        cnn_out_size = 64 * view_shape[1] * view_shape[2]

        # 2. Wspólne warstwy gęste (CNN + Goal Vector)
        self.fc = nn.Sequential(
            nn.Linear(cnn_out_size + goal_vec_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),  # Zwraca Q-values dla 5 akcji
        )

    def forward(self, view, goal_vec):
        cnn_features = self.cnn(view)
        # Łączymy cechy z obrazu z wektorem celu
        combined = torch.cat([cnn_features, goal_vec], dim=1)
        return self.fc(combined)


class DQNAgent:
    def __init__(self, model, epsilon=1.0, epsilon_min=0.01, decay=0.995):
        self.model = model
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.decay = decay
        self.n_actions = 5

    def get_action(self, obs, device="cpu"):
        # 1. Epsilon-greedy: Losowanie
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.n_actions)

        # 2. Wybór najlepszej akcji (Exploitation)
        # Przygotowanie danych (dodanie wymiaru batch)
        view = torch.FloatTensor(obs["view"]).unsqueeze(0).to(device)
        goal = torch.FloatTensor(obs["goal_vector"]).unsqueeze(0).to(device)

        with torch.no_grad():
            q_values = self.model(view, goal)

        return torch.argmax(q_values).item()

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.decay)


def optimize_model(batch, policy_net, target_net, optimizer, gamma):
    states, actions, rewards, next_states, dones = zip(*batch)

    device = next(policy_net.parameters()).device

    views = torch.stack([torch.FloatTensor(s["view"]) for s in states]).to(device)
    goal_vecs = torch.stack([torch.FloatTensor(s["goal_vector"]) for s in states]).to(
        device
    )

    actions = torch.LongTensor(actions).view(-1, 1).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    dones = torch.FloatTensor(dones).to(device)

    next_views = torch.stack([torch.FloatTensor(s["view"]) for s in next_states]).to(
        device
    )
    next_goal_vecs = torch.stack(
        [torch.FloatTensor(s["goal_vector"]) for s in next_states]
    ).to(device)

    current_q_values = policy_net(views, goal_vecs).gather(1, actions)

    with torch.no_grad():
        max_next_q_values = target_net(next_views, next_goal_vecs).max(1)[0]
        expected_q_values = rewards + (gamma * max_next_q_values * (1 - dones))

    loss = nn.MSELoss()(current_q_values, expected_q_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()

    # Opcjonalne: przycinanie gradientów (gradient clipping) zapobiega "wybuchaniu" wag
    # torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)

    optimizer.step()

    return loss.item()


def train(
    model_name: str = "DQN_model",
    view_size: int = 5,
    device: torch.device = torch.device("cpu"),
    num_episodes: int = 1000,
):
    # Inicjalizacja
    filename = f"{model_name}_{view_size}.pth"
    script_path = Path(__file__).parent
    dir_path = script_path.parent / "models"
    dir_path.mkdir(exist_ok=True)

    vshape = (4, view_size, view_size)
    policy_net = DQNet(view_shape=vshape, goal_vec_size=2, n_actions=5).to(device)
    target_net = DQNet(view_shape=vshape, goal_vec_size=2, n_actions=5).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-4)
    memory = ReplayBuffer(20000)
    batch_size = 512
    gamma = 0.99
    env = MultiRobotGridEnv(
        grid_size=(5, 5),
        num_agents=1,
        agent_view_size=view_size,
        step_limit=100,
    )
    agent_brain = DQNAgent(policy_net)

    completed_deliveries = [0] * num_episodes

    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False

        print(f"--- Epizode: {episode} ---")

        while not done:
            # 1. Wybierz akcje dla wszystkich agentów
            actions = {}
            tic = time.time()
            for agent_id, agent_obs in obs.items():
                actions[agent_id] = agent_brain.get_action(agent_obs, device)
            print(f"Action choose time: {time.time() - tic}")

            # 2. Wykonaj krok w środowisku
            tic = time.time()
            next_obs, rewards, terminated, truncated, _ = env.step(actions)
            print(f"Step time: {time.time() - tic}")

            if terminated:
                print("TERMINATED")
                completed_deliveries[episode] = 1
            if truncated:
                print("TRUNCATED")
                print(env.step_count)
            done = terminated or truncated

            # 3. Zapisz doświadczenie każdego agenta do pamięci
            tic = time.time()
            for agent_id in obs.keys():
                # if agent_id in next_obs:  # Tylko jeśli agent jeszcze istnieje
                if agent_id in next_obs:
                    memory.push(
                        obs[agent_id],
                        actions[agent_id],
                        rewards[agent_id],
                        next_obs[agent_id],
                        done,
                    )
            print(f"Memory push time: {time.time() - tic}")

            obs = next_obs

            tic = time.time()
            if len(memory) > batch_size:
                batch = memory.sample(batch_size)
                optimize_model(batch, policy_net, target_net, optimizer, gamma)
            print(f"Optimize time: {time.time() - tic}")

        # Co 10 epizodów aktualizuj Target Network
        if episode % 10 == 0:
            target_net.load_state_dict(policy_net.state_dict())
            torch.save(target_net.state_dict(), dir_path / filename)

    plt.plot(completed_deliveries)
    plt.show()


if __name__ == "__main__":
    print("Training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    train(num_episodes=500, device=device)
    print("Done")

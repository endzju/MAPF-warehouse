import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.core.MultiRobotGridEnv import MultiRobotGridEnv
from src.utils.plots import save_avg_stepcount, save_completed_deliveries_plot


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

    def get_action(self, obs: list[dict], device="cpu"):

        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.n_actions)

        view = torch.FloatTensor(obs["view"]).unsqueeze(0).to(device)
        goal = torch.FloatTensor(obs["goal_vector"]).unsqueeze(0).to(device)

        with torch.no_grad():
            q_values = self.model(view, goal)

        return torch.argmax(q_values).item()

    def get_actions(self, obs_list: list[dict], device="cpu"):

        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.n_actions)

        views = [o["view"] for o in obs_list]
        goals = [o["goal_vector"] for o in obs_list]

        views_tensor = torch.FloatTensor(np.stack(views)).to(device)
        goals_tensor = torch.FloatTensor(np.stack(goals)).to(device)

        with torch.no_grad():
            q_values = self.model(views_tensor, goals_tensor)
            actions = q_values.argmax(dim=1).cpu().numpy()

        return actions

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.decay)


def optimize_model(batch, policy_net, target_net, optimizer, gamma, scaler):
    states, actions, rewards, next_states, dones = zip(*batch)

    device = next(policy_net.parameters()).device
    goals_np = np.stack([s["goal_vector"] for s in states])
    goal_vecs = torch.from_numpy(goals_np).float().to(device, non_blocking=True)
    next_goals_np = np.stack([s["goal_vector"] for s in next_states])
    next_goal_vecs = (
        torch.from_numpy(next_goals_np).float().to(device, non_blocking=True)
    )
    views_np = np.stack([s["view"] for s in states])
    views = torch.from_numpy(views_np).float().to(device, non_blocking=True)
    next_views_np = np.stack([s["view"] for s in next_states])
    next_views = torch.from_numpy(next_views_np).float().to(device, non_blocking=True)

    actions = torch.LongTensor(actions).view(-1, 1).to(device, non_blocking=True)
    rewards = torch.FloatTensor(rewards).to(device, non_blocking=True)
    dones = torch.FloatTensor(dones).to(device, non_blocking=True)

    with torch.amp.autocast("cuda"):
        current_q_values = policy_net(views, goal_vecs).gather(1, actions)

        with torch.no_grad():
            max_next_q_values = target_net(next_views, next_goal_vecs).max(1)[0]
            expected_q_values = rewards + (gamma * max_next_q_values * (1 - dones))
            expected_q_values = expected_q_values.unsqueeze(1)

        loss = nn.MSELoss()(current_q_values, expected_q_values)

    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()

    return loss.item()


def train(
    num_episodes: int = 100,
    view_size: int = 5,
    num_agents: int = 1,
    env_shape: tuple[int, int] = (5, 5),
    step_limit: int = 100,
    task_length: int = 5,
    device: torch.device = torch.device("cpu"),
    out_model_name: str = "DQN_model",
    in_model_name=None,
    plot: bool = True,
    save_data: bool = False,
):
    # Inicjalizacja
    save_name = f"{out_model_name}_{view_size}"

    filename = f"{save_name}.pth"
    script_path = Path(__file__).parent
    model_path = script_path.parent / "models"
    plot_path = script_path.parent / "plots"
    model_path.mkdir(exist_ok=True)
    plot_path.mkdir(exist_ok=True)

    vshape = (4, view_size, view_size)
    policy_net = DQNet(view_shape=vshape, goal_vec_size=2, n_actions=5).to(device)
    target_net = DQNet(view_shape=vshape, goal_vec_size=2, n_actions=5).to(device)
    if in_model_name:
        path = model_path / f"{in_model_name}_{view_size}.pth"
        weights_dict = torch.load(path, map_location=device)
        policy_net.load_state_dict(weights_dict)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    memory = ReplayBuffer(20000)
    batch_size = 512
    gamma = 0.9
    env = MultiRobotGridEnv(
        grid_size=env_shape,
        num_agents=num_agents,
        agent_view_size=view_size,
        step_limit=step_limit,
        task_length=task_length,
    )
    agent_brain = DQNAgent(policy_net)

    completed_deliveries = [0] * num_episodes
    completion_steps = [env.step_limit] * num_episodes

    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False

        print(f"--- Epizode: {episode}, epsilon: {agent_brain.epsilon:.5f} ", end="")

        while not done:
            actions = {}
            for agent_id, agent_obs in obs.items():
                actions[agent_id] = agent_brain.get_action(agent_obs, device)

            next_obs, rewards, terminated, truncated, _ = env.step(actions)

            done = terminated or truncated
            if terminated:
                print("TERMINATED ---")
                completed_deliveries[episode] = 1
                completion_steps[episode] = env.step_count

            elif truncated:
                print("TRUNCATED ---")

            for agent_id in obs.keys():
                if agent_id in next_obs:
                    memory.push(
                        obs[agent_id],
                        actions[agent_id],
                        rewards[agent_id],
                        next_obs[agent_id],
                        done,
                    )

            obs = next_obs

            if len(memory) > batch_size:
                batch = memory.sample(batch_size)
                optimize_model(batch, policy_net, target_net, optimizer, gamma, scaler)

        agent_brain.update_epsilon()

        # Co 10 epizodów aktualizuj Target Network
        if episode % 10 == 0 and episode > 0:
            target_net.load_state_dict(policy_net.state_dict())
            torch.save(target_net.state_dict(), model_path / filename)

    if plot:
        save_completed_deliveries_plot(
            completed_deliveries,
            plot_path / f"{save_name}_completed_deliveries.png",
            save_data=save_data,
            window_size=20,
        )
        save_avg_stepcount(
            completion_steps,
            plot_path / f"{save_name}_avg_stepcount.png",
            save_data=save_data,
            window_size=20,
        )


if __name__ == "__main__":
    print("Training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    train(
        num_episodes=200,
        view_size=3,
        num_agents=3,
        env_shape=(5, 5),
        step_limit=80,
        task_length=5,
        device=device,
        out_model_name="DQN_model_triple3",
        # in_model_name="DQN_model_single8",
        plot=True,
        save_data=False,
    )
    print("Done")

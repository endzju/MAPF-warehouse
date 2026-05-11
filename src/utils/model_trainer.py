import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.agents.action_agent import ActionAgent
from src.core.MultiRobotGridEnv import MultiRobotGridEnv
from src.neural_networks.CNN.cnn import CNN1  # noqa: F401
from src.neural_networks.DQN.dqn import DQNet1, DQNet2, DQNet3  # noqa: F401
from src.utils.plots import (
    save_avg_stepcount,
    save_completed_deliveries_plot,
    save_stepcount,
)


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


def optimize_model(batch, policy_net, target_net, optimizer, gamma, scaler):
    states, actions, rewards, next_states, dones = zip(*batch)

    device = next(policy_net.parameters()).device

    batch_size = len(states)
    view_shape = states[0]["view"].shape
    goal_shape = states[0]["goal_vector"].shape

    # 2. Alokujemy pamięć w NumPy (bardzo szybkie)
    views_np = np.empty((batch_size, *view_shape), dtype=np.float32)
    next_views_np = np.empty((batch_size, *view_shape), dtype=np.float32)
    goals_np = np.empty((batch_size, *goal_shape), dtype=np.float32)
    next_goals_np = np.empty((batch_size, *goal_shape), dtype=np.float32)

    for i in range(batch_size):
        views_np[i] = states[i]["view"]
        next_views_np[i] = next_states[i]["view"]
        goals_np[i] = states[i]["goal_vector"]
        next_goals_np[i] = next_states[i]["goal_vector"]

    # 4. Konwersja na sensory PyTorch
    views = torch.as_tensor(views_np).to(device, non_blocking=True)
    next_views = torch.as_tensor(next_views_np).to(device, non_blocking=True)
    goal_vecs = torch.as_tensor(goals_np).to(device, non_blocking=True)
    next_goal_vecs = torch.as_tensor(next_goals_np).to(device, non_blocking=True)

    actions = (
        torch.as_tensor(actions, dtype=torch.long)
        .to(device, non_blocking=True)
        .view(-1, 1)
    )
    rewards = torch.as_tensor(rewards, dtype=torch.float32).to(
        device, non_blocking=True
    )
    dones = torch.as_tensor(dones, dtype=torch.float32).to(device, non_blocking=True)

    with torch.amp.autocast("cuda"):
        current_q_values = policy_net(views, goal_vecs).gather(1, actions)

        with torch.no_grad():
            max_next_q_values = target_net(next_views, next_goal_vecs).max(dim=1).values
            expected_q_values = rewards + (gamma * max_next_q_values * (1.0 - dones))
            expected_q_values = expected_q_values.unsqueeze(1)

        loss = F.mse_loss(current_q_values, expected_q_values)

    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()

    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)

    scaler.step(optimizer)
    scaler.update()

    return loss.item()


def train(
    env: MultiRobotGridEnv,
    model_class: nn.Module,
    num_episodes: int = 100,
    device: torch.device = torch.device("cpu"),
    in_model_name=None,
    out_model_name=None,
    plot: bool = True,
    save_data: bool = False,
    epsilon: float = 1.0,
    epsilon_min: float = 0.01,
    epsilon_decay: float = 0.995,
    epsilon_episodes: int = 500,
):
    if out_model_name is None:
        out_model_name = (
            f"{model_class.__name__}_{env.num_agents}_{env.agent_view_size}.pth"
        )
    view_size = env.agent_view_size

    filename = out_model_name
    nn_path = Path(__file__).parent.parent / "neural_networks"
    model_path = nn_path / "models"
    plot_path = nn_path / "plots"
    history_path = nn_path / "history"
    model_path.mkdir(exist_ok=True)
    plot_path.mkdir(exist_ok=True)
    history_path.mkdir(exist_ok=True)

    vshape = (4, view_size, view_size)
    policy_net = model_class(view_shape=vshape, goal_vec_size=2, n_actions=5).to(device)
    target_net = model_class(view_shape=vshape, goal_vec_size=2, n_actions=5).to(device)
    if in_model_name:
        path = model_path / in_model_name
        weights_dict = torch.load(path, map_location=device, weights_only=True)
        policy_net.load_state_dict(weights_dict)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    batch_size = 512
    num_batches = 1
    update_episodes = 20
    memory = ReplayBuffer(1000 * batch_size)

    gamma = 0.99
    agent_brain = ActionAgent(
        model=policy_net,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        decay=epsilon_decay,
    )

    completed_deliveries = [0] * num_episodes
    completion_steps = [env.step_limit] * num_episodes

    for episode in range(num_episodes):
        if episode > epsilon_episodes:
            agent_brain.epsilon = 0
        obs, _ = env.reset()
        done = False

        print(f"--- Episode: {episode}, epsilon: {agent_brain.epsilon:.5f} ", end="")

        while not done:
            actions = agent_brain.get_actions(obs, device)

            next_obs, rewards, terminated, truncated, _ = env.step(actions)

            done = terminated or truncated
            if terminated:
                print("SUCCESS ---")
                completed_deliveries[episode] = 1
                completion_steps[episode] = env.step_count

            elif truncated:
                print("TIMEOUT ---")

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
                for _ in range(num_batches):
                    batch = memory.sample(batch_size)
                    optimize_model(
                        batch, policy_net, target_net, optimizer, gamma, scaler
                    )

        agent_brain.update_epsilon()

        # Every {update_episodes} epizodes update Target Network and save model to history
        if episode % update_episodes == 0 and episode > 0:
            target_net.load_state_dict(policy_net.state_dict())
            torch.save(
                target_net.state_dict(), history_path / f"epizode{episode}_{filename}"
            )

    # Save model after training
    torch.save(policy_net.state_dict(), model_path / filename)

    if plot:
        save_completed_deliveries_plot(
            completed_deliveries=completed_deliveries,
            path=plot_path,
            filename=out_model_name,
            save_data=save_data,
            window_size=20,
            start_eps=epsilon,
            epsilon_decay=epsilon_decay,
        )
        save_avg_stepcount(
            completion_steps=completion_steps,
            path=plot_path,
            filename=out_model_name,
            save_data=save_data,
            window_size=20,
            start_eps=epsilon,
            epsilon_decay=epsilon_decay,
        )
        save_stepcount(
            completion_steps=completion_steps,
            path=plot_path,
            filename=out_model_name,
            save_data=save_data,
            start_eps=epsilon,
            epsilon_decay=epsilon_decay,
        )


if __name__ == "__main__":
    print("Training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    obs = [(1, 1), (3, 4)]
    env = MultiRobotGridEnv(
        grid_size=(10, 10),
        num_agents=8,
        agent_view_size=5,
        step_limit=300,
        task_length=5,
        # obstacles=obs,
    )
    train(
        num_episodes=50,
        env=env,
        device=device,
        out_model_name="CNN1+_8_5.pth",
        in_model_name="CNN1_8_5.pth",
        plot=True,
        save_data=False,
        model_class=CNN1,
        epsilon=0,
        epsilon_min=0,
        epsilon_decay=0.995,
    )
    print("Done")

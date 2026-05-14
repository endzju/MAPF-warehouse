import math
import random
import time
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


class EfficientReplayBuffer:
    def __init__(self, capacity, view_shape, goal_shape):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        self.views = np.zeros((capacity, *view_shape), dtype=np.float32)
        self.next_views = np.zeros((capacity, *view_shape), dtype=np.float32)
        self.goals = np.zeros((capacity, *goal_shape), dtype=np.float32)
        self.next_goals = np.zeros((capacity, *goal_shape), dtype=np.float32)

        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        idx = self.ptr
        self.views[idx] = state["view"]
        self.next_views[idx] = next_state["view"]
        self.goals[idx] = state["goal_vector"]
        self.next_goals[idx] = next_state["goal_vector"]
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (
            self.views[idxs],
            self.goals[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.next_views[idxs],
            self.next_goals[idxs],
            self.dones[idxs],
        )

    def __len__(self):
        return self.size


def optimize_model(batch, policy_net, target_net, optimizer, gamma, scaler):
    (
        views_np,
        goals_np,
        actions_np,
        rewards_np,
        next_views_np,
        next_goals_np,
        dones_np,
    ) = batch

    device = next(policy_net.parameters()).device

    views = torch.from_numpy(views_np).to(device, non_blocking=True)
    goal_vecs = torch.from_numpy(goals_np).to(device, non_blocking=True)
    actions = torch.from_numpy(actions_np).to(device, non_blocking=True).unsqueeze(1)
    rewards = torch.from_numpy(rewards_np).to(device, non_blocking=True)
    next_views = torch.from_numpy(next_views_np).to(device, non_blocking=True)
    next_goal_vecs = torch.from_numpy(next_goals_np).to(device, non_blocking=True)
    dones = torch.from_numpy(dones_np).to(device, non_blocking=True)

    with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
        current_q_values = policy_net(views, goal_vecs).gather(1, actions)

        with torch.no_grad():
            max_next_q_values = target_net(next_views, next_goal_vecs).max(dim=1).values
            expected_q_values = rewards + (gamma * max_next_q_values * (1.0 - dones))
            expected_q_values = expected_q_values.unsqueeze(1)

        loss = F.mse_loss(current_q_values, expected_q_values)

    optimizer.zero_grad(set_to_none=True)

    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
        optimizer.step()

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
    epsilon_episodes: int = math.inf,
    lr=1e-4,
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

    # if device.type == "cuda":
    #     policy_net = torch.compile(policy_net)

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    batch_size = 512 * 8
    num_batches = 50
    update_episodes = 20

    memory = EfficientReplayBuffer(
        capacity=1000 * batch_size,
        view_shape=vshape,
        goal_shape=(2,),
    )

    gamma = 0.99
    agent_brain = ActionAgent(
        model=policy_net,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        decay=epsilon_decay,
    )

    completed_deliveries = [0] * num_episodes
    completion_steps = [env.step_limit] * num_episodes

    tic = time.time()
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
                print("SUCCESS", end="")
                completed_deliveries[episode] = 1
                completion_steps[episode] = env.step_count

            elif truncated:
                print("TIMEOUT", end="")

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

        for _ in range(num_batches):
            batch = memory.sample(batch_size)
            optimize_model(batch, policy_net, target_net, optimizer, gamma, scaler)
        print(f" {time.time() - tic:.2f}s ---")
        tic = time.time()

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
        grid_size=(15, 15),
        num_agents=3,
        agent_view_size=7,
        step_limit=300,
        task_length=5,
        # obstacles=obs,
    )
    train(
        num_episodes=500,
        env=env,
        device=device,
        # out_model_name="CNN1+_8_5.pth",
        # in_model_name="CNN1_8_5.pth",
        plot=True,
        save_data=False,
        model_class=CNN1,
        # epsilon=0,
        # epsilon_min=0,
        # epsilon_decay=0.995,
        # epsilon_episodes=500
    )
    print("Done")

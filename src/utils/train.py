import copy
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
from src.utils.plots import plot_avg_completed_tasks_percentage, plot_avg_stepcount


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


def get_window_avg(sequence, window_size):
    window_avg = []
    for i in range(len(sequence) // window_size):
        task_window = sequence[i * window_size : (i + 1) * window_size]
        window_avg.append(sum(task_window) / len(task_window))
    return window_avg


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
    env_grid_size: tuple[int, int],
    env_max_robots: int,
    env_agent_view_size: int,
    env_step_limit: int,
    env_task_length: int,
    env_num_tasks: int,
    model: nn.Module,
    num_episodes: int,
    epsilon: float,
    epsilon_min: float,
    epsilon_decay: float,
    epsilon_episodes: int,
    num_batches: int,
    device: torch.device = torch.device("cpu"),
    plot: bool = True,
    save_plot_data: bool = False,
    lr=1e-4,
) -> tuple[list[nn.Module], list[float], list[float]]:
    """
    Train a model.

    Returns:
        list[nn.Module]: List of trained models.
        list[float]: List of completed tasks per episode.
        list[float]: List of steps per episode.

    """
    env = MultiRobotGridEnv(
        grid_size=env_grid_size,
        max_robots=env_max_robots,
        agent_view_size=env_agent_view_size,
        step_limit=env_step_limit,
        task_length=env_task_length,
        num_tasks=env_num_tasks,
    )

    view_size = env.agent_view_size
    nn_path = Path(__file__).parent.parent / "neural_networks"
    plot_path = nn_path / "plots" / f"{model.display_name}"
    plot_path.mkdir(exist_ok=True)

    vshape = (4, view_size, view_size)

    policy_net = model.to(device)
    target_net = copy.deepcopy(model).to(device)

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    batch_size = 512 * 8
    update_episodes = 10

    memory = EfficientReplayBuffer(
        capacity=500 * batch_size,
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

    completed_tasks = [0] * num_episodes
    completion_steps = [env.step_limit] * num_episodes

    model_history = []

    tic = time.time()
    for episode in range(num_episodes):
        if episode > epsilon_episodes:
            agent_brain.epsilon = 0
        obs, _ = env.reset()
        done = False

        print(f"--- Episode: {episode}, epsilon: {agent_brain.epsilon:.5f} ", end="")

        while not done:
            actions = agent_brain.get_actions(obs_dict=obs, device=device)
            next_obs, rewards, terminated, truncated, _ = env.step(actions)
            done = terminated or truncated
            if done:
                message = "TIMEOUT" if truncated else "SUCCESS"
                print(
                    f"{message}, tasks completed: {env.get_num_tasks_completed()}/{len(env.tasks)}",
                    end="",
                )
                completed_tasks[episode] = env.get_num_tasks_completed()
                completion_steps[episode] = env.step_count

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
        print(f" {time.time() - tic:.2f}s ---", end="\r")
        tic = time.time()
        agent_brain.update_epsilon()

        # Every {update_episodes} episodes update Target Network and add model to history
        if (episode + 1) % update_episodes == 0:
            model_history.append(copy.deepcopy(policy_net).cpu())
            target_net.load_state_dict(policy_net.state_dict())
    print()
    filename = (
        f"{model.display_name}_b{num_batches}_r{env_max_robots}_v{env_agent_view_size}"
    )
    avg_completed_tasks = get_window_avg(completed_tasks, update_episodes)
    avg_completion_steps = get_window_avg(completion_steps, update_episodes)

    if plot:
        plot_avg_completed_tasks_percentage(
            avg_completed_tasks=avg_completed_tasks,
            max_tasks=env.num_tasks,
            path=plot_path,
            filename=filename,
            save_plot_data=save_plot_data,
            window_size=update_episodes,
        )
        plot_avg_stepcount(
            avg_completion_steps=avg_completion_steps,
            path=plot_path,
            filename=filename,
            save_plot_data=save_plot_data,
            window_size=update_episodes,
        )

    return model_history, avg_completed_tasks, avg_completion_steps


def run_training(
    model: nn.Module,
    num_robots: int,
    view_size: int,
    num_batches: int,
    params: dict,
) -> list[tuple[list[nn.Module], list[int], list[int]]]:
    """
    Runs models training.

    Returns:
        list[nn.Module]: List of trained models.
        list[float]: List of completed tasks per episode.
        list[float]: List of steps per episode.

    """
    params = params.copy()
    params["model"] = model
    params["env_max_robots"] = num_robots
    params["env_num_tasks"] = num_robots * 5
    params["env_agent_view_size"] = view_size
    params["num_batches"] = num_batches

    return train(**params)


if __name__ == "__main__":
    pass

import copy
import random
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from src.agents.action_agent import ActionAgent
from src.core.MultiRobotGridEnv import MultiRobotGridEnv
from src.neural_networks.model_config import ModelConfig
from src.utils.plots import plot_avg_delivery_times


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
    def __init__(self, capacity, view_shape, additional_input_shape):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        self.views = np.zeros((capacity, *view_shape), dtype=np.float32)
        self.next_views = np.zeros((capacity, *view_shape), dtype=np.float32)
        self.goals = np.zeros((capacity, *additional_input_shape), dtype=np.float32)
        self.next_goals = np.zeros(
            (capacity, *additional_input_shape), dtype=np.float32
        )

        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        idx = self.ptr
        self.views[idx] = state["view"]
        self.next_views[idx] = next_state["view"]
        self.goals[idx] = state["additional_input"]
        self.next_goals[idx] = next_state["additional_input"]
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
    model_config: ModelConfig,
    env: MultiRobotGridEnv,
    # Train params:
    num_episodes: int,
    device: torch.device,
    epsilon: float,
    epsilon_min: float,
    epsilon_decay: float,
    epsilon_episodes: int,
    best_model_window: int,
    # Config params:
    batch_size: int,
    num_batches: int,
    buffer_length: int,
    target_update_interval: int,
    gamma: float,
    # Other params:
    verbose: int = 0,
    plot: bool = True,
    lr=1e-4,
) -> tuple[list[nn.Module], list[float], list[float]]:
    """
    Train a model.

    Returns:
        list[nn.Module]: List of trained models.
        list[float]: List of completed tasks per episode.
        list[float]: List of steps per episode.

    """

    nn_path = Path(__file__).parent.parent / "neural_networks"
    plot_path = nn_path / "plots" / model_config.get_model_dir_name()
    plot_path.mkdir(exist_ok=True, parents=True)

    model = model_config.build_model()
    policy_net = model.to(device)
    target_net = copy.deepcopy(model).to(device)
    policy_net.train()
    target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    memory = EfficientReplayBuffer(
        capacity=buffer_length,
        view_shape=model_config.get_view_input_shape(),
        additional_input_shape=(model_config.get_additional_input_size(),),
    )

    agent_brain = ActionAgent(
        model=policy_net,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        decay=epsilon_decay,
    )

    completed_tasks = [0] * num_episodes
    avg_delivery_times = [0] * num_episodes
    avg_manhattan_times = [0] * num_episodes

    best_model = None
    best_completed_tasks = -1
    best_ratio = -1

    tic = time.time()
    for episode in range(num_episodes):
        if episode > epsilon_episodes:
            agent_brain.epsilon = 0
        obs, _ = env.reset()
        done = False
        if verbose > 0:
            print(f"-Episode: {episode}, epsilon: {agent_brain.epsilon:.5f}", end="")
        simulation_tic = time.time()
        action_time = 0
        env_step_time = 0
        memory_push_time = 0
        policy_net.to("cpu")
        while not done:
            action_tic = time.time()
            actions = agent_brain.get_actions(obs_dict=obs, device="cpu")
            action_time += time.time() - action_tic
            step_tic = time.time()
            next_obs, rewards, terminated, truncated, _ = env.step(actions)
            env_step_time += time.time() - step_tic
            done = terminated or truncated
            if done:
                if verbose > 0:
                    print(
                        f", tasks completed: {env.get_num_tasks_completed()}/{len(env.tasks)}",
                        end="",
                    )
                tasks_completed = env.get_num_tasks_completed()
                completed_tasks[episode] = tasks_completed
                avg_delivery_times[episode] = (
                    tasks_completed * env.avg_delivery_time
                    + (len(env.tasks) - tasks_completed) * env.step_limit
                ) / len(env.tasks)
                avg_manhattan_times[episode] = env.avg_manhattan_distance

            memory_push_tic = time.time()
            for agent_id in obs:
                if agent_id in next_obs:
                    memory.push(
                        obs[agent_id],
                        actions[agent_id],
                        rewards[agent_id],
                        next_obs[agent_id],
                        done,
                    )
            memory_push_time += time.time() - memory_push_tic
            obs = next_obs
        if verbose > 0:
            print(f", simulation: {time.time() - simulation_tic:.2f}s", end="")
        if verbose > 1:
            print(f"(action: {action_time:.2f}s", end="")
            print(f", env step: {env_step_time:.2f}s", end="")
            print(f", memory push: {memory_push_time:.2f}s)", end="")
        optimize_tic = time.time()
        sample_time = 0
        policy_net.to(device)
        for _ in range(num_batches):
            sample_tic = time.time()
            batch = memory.sample(batch_size)
            sample_time += time.time() - sample_tic
            optimize_model(batch, policy_net, target_net, optimizer, gamma, scaler)
        if verbose > 0:
            print(f", optimize: {time.time() - optimize_tic:.2f}s", end="")
            print(f", episode time: {time.time() - tic:.2f}s-", end="\r")
        tic = time.time()
        agent_brain.update_epsilon()

        # Every {best_model_window} episodes check if trained model is new best model
        if (episode + 1) % best_model_window == 0:
            start = episode - best_model_window + 1
            stop = episode + 1
            sum_delivery_time = sum(avg_delivery_times[start:stop])
            sum_manhattan_distance = sum(avg_manhattan_times[start:stop])
            avg_window_completed_tasks = (
                sum(completed_tasks[start:stop]) / best_model_window
            )
            ratio = sum_manhattan_distance / (sum_delivery_time + 0.0001)
            if (
                avg_window_completed_tasks > best_completed_tasks
                or avg_window_completed_tasks == best_completed_tasks
                and ratio > best_ratio
            ):
                best_model = copy.deepcopy(policy_net).cpu()
                best_completed_tasks = avg_window_completed_tasks
                best_ratio = ratio

        # Every {target_update_interval} episodes update Target Network and add model to history
        if (episode + 1) % target_update_interval == 0:
            target_net.load_state_dict(policy_net.state_dict())
    print()

    avg_completed_tasks = get_window_avg(completed_tasks, best_model_window)
    window_avg_delivery_times = get_window_avg(avg_delivery_times, best_model_window)
    window_avg_manhattan_times = get_window_avg(avg_manhattan_times, best_model_window)

    if plot:
        plot_avg_delivery_times(
            avg_delivery_times=window_avg_delivery_times,
            avg_manhattan_times=window_avg_manhattan_times,
            path=plot_path,
            checkpoint_name=model.checkpoint_name,
            window_size=best_model_window,
        )

    return (
        best_model,
        avg_completed_tasks,
        window_avg_delivery_times,
        window_avg_manhattan_times,
    )


def _train_worker(args: tuple) -> ModelConfig:
    config, env_params, train_params, force_train = args
    should_train = force_train or not config.get_model_path().exists()
    if not should_train:
        print(f"Model {config.get_model_full_name()} already exists. Skipping...")
        return config

    env = MultiRobotGridEnv(
        **env_params,
        **config.get_env_params(),
    )
    train_results = train(
        env=env,
        model_config=config,
        **train_params,
        **config.get_train_params(),
        verbose=0,
    )
    best_trained_model, _, _, _ = train_results
    model_path = config.get_model_path()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_trained_model.state_dict(), model_path)
    return config


def run_training(
    model_configs: list[ModelConfig],
    env_base_params: dict,
    train_base_params: dict,
    force_train: bool,
    num_processes: int,
) -> list[tuple[list[nn.Module], list[int], list[int]]]:
    """
    Runs models training.
    """
    tasks = [
        (config, env_base_params, train_base_params, force_train)
        for config in model_configs
    ]
    trained_configs = []

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [executor.submit(_train_worker, task) for task in tasks]
        for future in tqdm(
            as_completed(futures), total=len(tasks), desc="Training models"
        ):
            completed_config = future.result()
            trained_configs.append(completed_config)


if __name__ == "__main__":
    pass

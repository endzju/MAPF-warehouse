import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import mean

from filelock import FileLock
from torch import nn
from tqdm import tqdm

from src.agents.action_agent import ActionAgent
from src.core.MultiRobotGridEnv import MultiRobotGridEnv
from src.neural_networks.model_config import ModelConfig


def run_simulation(
    model: nn.Module,
    env: MultiRobotGridEnv,
    render=False,
):
    observations, _ = env.reset()
    if render:
        env.render()
    terminated = False
    truncated = False
    dqn_agent = ActionAgent(model, epsilon=0, epsilon_min=0, decay=0)
    while not (terminated or truncated):
        # 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT, 4=WAIT
        actions = dqn_agent.get_actions(observations, device="cpu")
        observations, _, terminated, truncated, _ = env.step(actions)

    return env.avg_manhattan_distance, env.avg_delivery_time


def safe_write_json(data: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = Path(__file__).parent.parent / "data" / "eval" / "~global.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(lock_path, timeout=30)

    with lock:
        try:
            with open(path, "r", encoding="utf-8") as f:
                old_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            old_data = {}
        old_data.update(data)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(old_data, f, indent=4)


def evaluate(
    num_simulations: int,
    env: MultiRobotGridEnv,
    model: nn.Module,
) -> dict:
    """
    Evaluates model, measure delivery time of random order placements.
    """
    manhattan_delivery_times = []
    delivery_times = []
    model.eval()

    for _ in range(num_simulations):
        avg_manhattan_delivery_time, avg_delivery_time = run_simulation(
            model=model,
            env=env,
            render=False,
        )
        manhattan_delivery_times.append(avg_manhattan_delivery_time)
        delivery_times.append(avg_delivery_time)

    movement_efficiency = sum(delivery_times) / sum(manhattan_delivery_times)
    robot_throughput_per_100ticks = 100 / mean(delivery_times) * env.num_robots
    eval_results = {
        "manhattan_delivery_times": manhattan_delivery_times,
        "avg_manhattan_delivery_time": mean(manhattan_delivery_times),
        "delivery_times": delivery_times,
        "avg_delivery_time": mean(delivery_times),
        "movement_efficiency": movement_efficiency,
        "robot_throughput_per_100ticks": robot_throughput_per_100ticks,
        "view_size": env.agent_view_size,
    }
    return eval_results


def _eval_worker(args: tuple) -> ModelConfig:
    config, env_params, num_simulations, data_path = args
    model = config.load_model()
    env = MultiRobotGridEnv(
        **env_params,
    )
    eval_results = evaluate(
        num_simulations=num_simulations,
        env=env,
        model=model,
    )
    data = {env_params["env_max_robots"]: eval_results}
    safe_write_json(data, data_path)
    return config


def run_evaluation(
    model_configs: list[ModelConfig],
    env_base_params: dict,
    eval_robot_list: list[int],
    num_simulations: int,
    num_processes: int,
) -> None:

    # tic = time.time()
    # print(f"Evaluating model: {model.model_name}_{model.checkpoint_name}")

    # elapsed = time.time() - tic
    # print(f"Evaluation time: {elapsed / 60:.1f} min")

    tasks = []
    for model_config in model_configs:
        data_path = (
            Path(__file__).parent.parent
            / "data"
            / "eval"
            / model_config.get_model_dir_name()
            / f"{model_config.get_params_string()}.json"
        )
        missing_num_robots = eval_robot_list
        if data_path.exists():
            with open(data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            keys = map(int, data.keys())
            missing_num_robots = list(set(eval_robot_list) - set(keys))
        env_params = env_base_params.copy()
        env_params.update(model_config.get_env_params())
        env_params["env_step_limit"] = 50000
        for num_robots in missing_num_robots:
            task_env_params = env_params.copy()
            task_env_params["env_max_robots"] = num_robots
            task_env_params["env_num_tasks"] = num_robots * 5
            tasks.append((model_config, task_env_params, num_simulations, data_path))

    eval_configs = []

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [executor.submit(_eval_worker, task) for task in tasks]
        for future in tqdm(
            as_completed(futures), total=len(tasks), desc="Evaluating models"
        ):
            completed_config = future.result()
            eval_configs.append(completed_config)


if __name__ == "__main__":
    pass

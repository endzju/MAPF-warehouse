import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import mean

from torch import nn
from tqdm import tqdm

from src.agents.action_agent import ActionAgent
from src.core.MultiRobotGridEnv import MultiRobotGridEnv


def run_simulation(
    model: nn.Module,
    env: MultiRobotGridEnv,
    render=False,
):
    observations, info = env.reset()
    if render:
        env.render()

    terminated = False
    truncated = False

    dqn_agent = ActionAgent(model, epsilon=0, epsilon_min=0, decay=0)
    while not (terminated or truncated):
        # 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT, 4=WAIT

        actions = dqn_agent.get_actions(observations, device="cpu")
        observations, rewards, terminated, truncated, info = env.step(actions)

    return env.avg_manhattan_distance, env.avg_delivery_time


def evaluate(
    num_simulations: int,
    env_grid_size: tuple[int, int],
    env_max_robots: int,
    env_agent_view_size: int,
    env_step_limit: int,
    env_task_length: int,
    env_num_tasks: int,
    model_state: dict,
    model_class: type,
    model_params: dict,
    **kwargs,
) -> dict:
    """
    Evaluates model, measure delivery time of random order placements.
    """
    env = MultiRobotGridEnv(
        grid_size=env_grid_size,
        agent_view_size=env_agent_view_size,
        max_robots=env_max_robots,
        step_limit=env_step_limit,
        task_length=env_task_length,
        num_tasks=env_num_tasks,
    )
    manhattan_delivery_times = []
    delivery_times = []

    model = model_class(**model_params, view_size=env_agent_view_size).cpu()
    model.load_state_dict(model_state)
    model.eval()

    for i in range(num_simulations):
        avg_manhattan_delivery_time, avg_delivery_time = run_simulation(
            model=model,
            env=env,
            render=False,
        )
        manhattan_delivery_times.append(avg_manhattan_delivery_time)
        delivery_times.append(avg_delivery_time)

    movement_efficiency = sum(delivery_times) / sum(manhattan_delivery_times)
    robot_throughput_per_100ticks = 100 / mean(delivery_times) * env_max_robots
    data = {
        "manhattan_delivery_times": manhattan_delivery_times,
        "avg_manhattan_delivery_time": mean(manhattan_delivery_times),
        "delivery_times": delivery_times,
        "avg_delivery_time": mean(delivery_times),
        "movement_efficiency": movement_efficiency,
        "robot_throughput_per_100ticks": robot_throughput_per_100ticks,
        "view_size": env.agent_view_size,
    }
    return data


def run_eval_task(params: dict) -> dict:
    return evaluate(**params)


def run_evaluation(
    model: nn.Module,
    num_robot_list: list[int],
    train_num_robots: int,
    num_batches: int,
    update_episodes: int,
    params: dict,
    num_simulations: int,
    num_processes: int,
) -> None:

    data_path = Path(__file__).parent.parent / "data" / "times" / model.display_name
    filename = f"{model.display_name}_b{num_batches}_r{train_num_robots}_v{model.view_size}_u{update_episodes}.json"
    full_path = data_path / filename
    if full_path.exists():
        return

    params = params.copy()

    params["model_state"] = model.state_dict()
    params["model_class"] = model.__class__
    params["env_agent_view_size"] = model.view_size
    params["env_step_limit"] = 50000
    params["num_simulations"] = num_simulations
    params["model_params"] = {
        "hidden_layers": model.hidden_layers,
        "input_size": model.input_size,
        "output_size": model.output_size,
    }

    tasks = []
    for num_robots in reversed(num_robot_list):
        params["env_max_robots"] = num_robots
        params["env_num_tasks"] = num_robots * 5
        tasks.append(params.copy())

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = {
            executor.submit(run_eval_task, task): i for i, task in enumerate(tasks)
        }

        result = [None] * len(tasks)

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Evaluating",
        ):
            idx = futures[future]
            result[idx] = future.result()

    result = reversed(result)

    data_path.mkdir(parents=True, exist_ok=True)
    data = {
        num_robots: stats
        for num_robots, stats in zip(num_robot_list, result, strict=True)
    }

    if full_path.exists():
        old_data = json.loads(full_path.read_text(encoding="utf-8"))
        data = {**old_data, **data}
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    pass

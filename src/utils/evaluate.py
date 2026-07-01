import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import mean

from torch import nn

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
    paused = False

    quit_requested, pause_pressed = False, False
    dqn_agent = ActionAgent(model, epsilon=0, epsilon_min=0, decay=0)
    while not (terminated or truncated):
        if render:
            quit_requested, pause_pressed = env.handle_events()

        if quit_requested:
            break

        if pause_pressed:
            paused = not paused

        if paused and render:
            env.render(paused=True)
            time.sleep(0.1)
            continue

        # 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT, 4=WAIT
        actions = {
            agent_id: dqn_agent.get_action(obs, device="cpu")
            for agent_id, obs in observations.items()
        }

        observations, rewards, terminated, truncated, info = env.step(actions)
        if render:
            env.render()
            time.sleep(0.1)

    if render:
        time.sleep(1)
    return env.avg_manhattan_distance, env.avg_delivery_time


def evaluate(
    env_grid_size: tuple[int, int],
    env_max_robots: int,
    env_agent_view_size: int,
    env_step_limit: int,
    env_task_length: int,
    env_num_tasks: int,
    model_state: dict,
    model_class: type,
    num_simulations: int,
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
    manhatan_delivery_times = []
    delivery_times = []

    vshape = (4, env_agent_view_size, env_agent_view_size)
    model = model_class(view_shape=vshape, goal_vec_size=2, n_actions=5)
    model.load_state_dict(model_state)
    model.eval()

    for i in range(num_simulations):
        avg_manhatan_delivery_time, avg_delivery_time = run_simulation(
            model=model,
            env=env,
            render=False,
        )
        manhatan_delivery_times.append(avg_manhatan_delivery_time)
        delivery_times.append(avg_delivery_time)

    movement_efficiency = sum(delivery_times) / sum(manhatan_delivery_times)
    robot_throughput_per_100ticks = 100 / mean(delivery_times) * env_max_robots
    data = {
        "manhatan_delivery_times": manhatan_delivery_times,
        "avg_manhatan_delivery_time": mean(manhatan_delivery_times),
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
    view_size: int,
    params: dict,
    is_tuned: bool = False,
    num_simulations: int = 10,
) -> None:
    params = params.copy()

    params["model_state"] = model.state_dict()
    params["model_class"] = model.__class__
    params["env_agent_view_size"] = view_size
    params["env_step_limit"] = 50000
    params["num_simulations"] = num_simulations

    tasks = []
    for num_robots in reversed(num_robot_list):
        params["env_max_robots"] = num_robots
        params["env_num_tasks"] = num_robots * 5
        tasks.append(params.copy())

    with ProcessPoolExecutor(max_workers=3) as executor:
        result = list(executor.map(run_eval_task, tasks))
    result = reversed(result)
    data_path = (
        Path(__file__).parent.parent / "data" / "times" / model.__class__.__name__
    )
    data_path.mkdir(parents=True, exist_ok=True)
    data = {
        num_robots: stats
        for num_robots, stats in zip(num_robot_list, result, strict=True)
    }
    infix = "_tuned" if is_tuned else ""
    filename = f"{model.__class__.__name__}{infix}_robots{train_num_robots}_view{view_size}.json"
    full_path = data_path / filename
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    pass

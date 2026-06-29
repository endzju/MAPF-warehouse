import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from src.agents.action_agent import ActionAgent
from src.core.MultiRobotGridEnv import MultiRobotGridEnv
from src.models.depot import Depot
from src.neural_networks.CNN.cnn import CNN1  # noqa: F401
from src.neural_networks.MLP.mlp import MLP1, MLP2, MLP3  # noqa: F401
from utils.load import load_model


def run_simulation(
    model_path="DQN_model_5.pth",
    env: MultiRobotGridEnv = None,
    model_class: MLP1 | MLP2 | MLP3 = MLP1,
    render=True,
):

    if ".pth" not in model_path:
        model_path += ".pth"

    observations, info = env.reset()
    if render:
        env.render()

    terminated = False
    truncated = False
    total_step = 0
    paused = False

    while not (terminated or truncated):
        quit_requested, pause_pressed = False, False
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
        model = load_model(model_path, model_class=model_class)
        model.eval()
        dqn_agent = ActionAgent(model, epsilon=0, epsilon_min=0, decay=0)

        actions = {
            agent_id: dqn_agent.get_action(obs, device="cpu")
            for agent_id, obs in observations.items()
        }

        observations, rewards, terminated, truncated, info = env.step(actions)

        total_step += 1
        if render:
            env.render()
            time.sleep(0.1)

    if render:
        time.sleep(1)
    return env.avg_manhattan_distance, env.avg_delivery_time


def run_evaluation(
    env, num_robots, model_path, model_class, num_simulations=10
) -> None:
    """
    Evaluates model, measure delivery time of random order placements.
    """
    manhatan_delivery_times = []
    delivery_times = []
    env.max_robots = num_robots
    env.num_tasks = 5 * num_robots
    data_path = Path(__file__).parent.parent / "data" / "times"
    data_path.mkdir(parents=True, exist_ok=True)

    for i in range(num_simulations):
        avg_manhatan_delivery_time, avg_delivery_time = run_simulation(
            model_path=model_path,
            env=env,
            model_class=model_class,
            render=False,
        )
        manhatan_delivery_times.append(avg_manhatan_delivery_time)
        delivery_times.append(avg_delivery_time)
    name = f"{model_class.__name__}_{num_robots}_{env.agent_view_size}"
    file_path = data_path / f"{name}.json"
    longer_delivery = sum(delivery_times) / sum(manhatan_delivery_times)
    robot_throughput_per_100ticks = (
        100 / (sum(delivery_times) / len(delivery_times)) * num_robots
    )
    data = {
        "manhatan_delivery_times": manhatan_delivery_times,
        "delivery_times": delivery_times,
        "longer_delivery": longer_delivery,
        "robot_throughput_per_100ticks": robot_throughput_per_100ticks,
        "num_robots": num_robots,
        "view_size": env.agent_view_size,
    }
    with file_path.open("w") as f:
        json.dump(data, f)


def run_task(args):
    model_class, model_path, num_robots = args

    depot1 = Depot((0, 0))
    depot2 = Depot((19, 0))
    depot3 = Depot((0, 19))
    depot4 = Depot((19, 19))
    env = MultiRobotGridEnv(
        grid_size=(20, 20),
        agent_view_size=7,
        obstacles=None,
        input_depots=[depot1, depot2],
        output_depots=[depot3, depot4],
        step_limit=5000,
        task_length=5,
    )

    return run_evaluation(
        env=env,
        num_robots=num_robots,
        model_path=model_path,
        model_class=model_class,
    )


if __name__ == "__main__":
    model_classes = [
        MLP2,
    ]
    model_paths = [
        "final_DQNet2+_50_7.pth",
    ]

    tasks = [
        (model_class, model_path, num_robots)
        for model_class, model_path in zip(model_classes, model_paths)
        for num_robots in range(10, 81, 10)
    ]

    with ProcessPoolExecutor(max_workers=4) as executor:
        list(executor.map(run_task, tasks))

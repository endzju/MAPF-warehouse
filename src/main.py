import time
from pathlib import Path

import torch

from src.agents.action_agent import ActionAgent
from src.core.MultiRobotGridEnv import MultiRobotGridEnv
from src.models.depot import Depot
from src.neural_networks.MLP.mlp import MLP


def main(
    model: Path,
    env: MultiRobotGridEnv,
    render=True,
):

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
        model.cpu().eval()
        dqn_agent = ActionAgent(model, epsilon=0, epsilon_min=0, decay=0)

        actions = {
            agent_id: dqn_agent.get_action(obs, device="cpu")
            for agent_id, obs in observations.items()
        }

        observations, rewards, terminated, truncated, info = env.step(actions)

        total_step += 1
        if render:
            env.render()
            time.sleep(0.07)

    if render:
        time.sleep(1)
    return env.avg_manhattan_distance, env.avg_delivery_time


if __name__ == "__main__":
    depot1 = Depot((0, 0))
    depot2 = Depot((19, 0))
    depot3 = Depot((0, 19))
    depot4 = Depot((19, 19))
    env = MultiRobotGridEnv(
        grid_size=(20, 20),
        obstacles=None,
        input_depots=[depot1, depot2],
        output_depots=[depot3, depot4],
        step_limit=5000,
        task_length=5,
        max_robots=20,
        num_tasks=100,
    )
    models_path = Path(__file__).resolve().parent / "neural_networks" / "models"

    # CONFIG
    model_name = "MLP_512_b120_r60_v7_u60.pth"
    model_class = MLP

    model_dir = model_name.split("_b")[0]
    model_path = models_path / model_dir / model_name
    hidden_layers = list(map(int, model_dir.split("_")[1:]))
    view_size = int(model_name.split("_v")[1].split("_")[0])
    vshape = (4, view_size, view_size)
    env.agent_view_size = view_size
    goal_vec_size = 2
    n_actions = 5
    input_size = vshape[0] * vshape[1] * vshape[2] + goal_vec_size
    model = model_class(
        input_size=input_size, hidden_layers=hidden_layers, output_size=n_actions
    ).to("cpu")
    weights_dict = torch.load(model_path, weights_only=True)
    model.load_state_dict(weights_dict)

    try:
        num_robots = 60
        manhattan_delivery_times = []
        delivery_times = []
        env.max_robots = num_robots
        env.num_tasks = 500 * num_robots
        avg_manhattan_delivery_time, avg_delivery_time = main(
            model=model,
            env=env,
            render=True,
        )
        print()

    except KeyboardInterrupt:
        print("Przerwano program")

    except Exception as e:
        print(e)

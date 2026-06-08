import json
import time
from pathlib import Path

from src.agents.action_agent import ActionAgent
from src.core.MultiRobotGridEnv import MultiRobotGridEnv
from src.models.depot import Depot
from src.neural_networks.CNN.cnn import CNN1  # noqa: F401
from src.neural_networks.DQN.dqn import DQNet1, DQNet2, DQNet3
from src.utils.model_loader import load_model


def main(
    model_path="DQN_model_5.pth",
    env: MultiRobotGridEnv = None,
    model_class: DQNet1 | DQNet2 | DQNet3 = DQNet1,
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
            for agent_id, obs in enumerate(observations)
        }

        observations, rewards, terminated, truncated, info = env.step(actions)

        total_step += 1
        if render:
            env.render()
            time.sleep(0.1)
    print("Symulacja zakończona")
    if render:
        time.sleep(1)
    return env.avg_manhattan_distance, env.avg_delivery_time


if __name__ == "__main__":
    model_path = "CNN1+copy_8_5.pth"
    # model_path = "DQNet2+_8_5.pth"
    model_path = "CNN1_30_5.pth"

    depot1 = Depot((0, 0))
    depot2 = Depot((19, 0))
    depot3 = Depot((0, 19))
    depot4 = Depot((19, 19))
    env = MultiRobotGridEnv(
        grid_size=(20, 20),
        agent_view_size=5,
        obstacles=None,
        input_depots=[depot1, depot2],
        output_depots=[depot3, depot4],
        step_limit=5000,
        task_length=5,
        max_robots=100,
        num_tasks=50,
    )
    model_classes = [CNN1]

    data_path = Path(__file__).parent / "data" / "times"
    data_path.mkdir(parents=True, exist_ok=True)

    try:
        for model_class in model_classes:
            manhatan_delivery_times = []
            delivery_times = []
            for num_robots in range(10, 101, 10):
                env.max_robots = 5
                env.reset()
                avg_manhatan_delivery_time, avg_delivery_time = main(
                    model_path=model_path,
                    env=env,
                    model_class=model_class,
                    render=True,
                )
                manhatan_delivery_times.append(avg_manhatan_delivery_time)
                delivery_times.append(avg_delivery_time)
            name = f"{model_class.__name__}_{env.max_robots}_{env.agent_view_size}"
            file_path = data_path / f"{name}.json"
            data = {
                "manhatan_delivery_times": manhatan_delivery_times,
                "delivery_times": delivery_times,
            }
            with file_path.open("w") as f:
                json.dump(data, f)

    except KeyboardInterrupt:
        env.close()
        print("Przerwano program")

import time

from src.agents.action_agent import ActionAgent
from src.core.MultiRobotGridEnv import MultiRobotGridEnv
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

    print("Symulacja zakończona")
    time.sleep(1)
    return env.avg_manhattan_distance, env.avg_delivery_time


if __name__ == "__main__":
    model_path = "CNN1+copy_8_5.pth"
    # model_path = "DQNet2+_8_5.pth"
    model_path = "CNN1_30_5.pth"
    # env = MultiRobotGridEnv(
    #     grid_size=(50, 50),
    #     num_agents=50,
    #     agent_view_size=5,
    #     step_limit=5000,
    # )
    env = MultiRobotGridEnv(
        grid_size=(20, 20),
        num_agents=30,
        agent_view_size=5,
        step_limit=5000,
    )

    total_manhatan_delivery_time = 0
    total_delivery_time = 0
    try:
        for i in range(10):
            manhatan_delivery_time, delivery_time = main(
                model_path=model_path,
                env=env,
                model_class=CNN1,
            )
            total_manhatan_delivery_time += manhatan_delivery_time
            total_delivery_time += delivery_time
    except KeyboardInterrupt:
        env.close()
        print("Przerwano program")

    print(f"Avarage manhatan delivery time: {total_manhatan_delivery_time / 10}")
    print(f"Avarage delivery time: {total_delivery_time / 10}")

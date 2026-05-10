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
):

    if ".pth" not in model_path:
        model_path += ".pth"

    observations, info = env.reset()

    terminated = False
    truncated = False
    total_step = 0

    while not (terminated or truncated):
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

        print(f"--- Krok: {total_step} ---")
        print(f"Nagrody: {rewards}")
        for agent in env.agents:
            print(f"{agent.id} {agent.task_type}", end=" ")
        print()
        env.render()

        time.sleep(0.1)

    print("Symulacja zakończona")


if __name__ == "__main__":
    model_path = "CNN1+copy_8_5.pth"
    env = MultiRobotGridEnv(
        grid_size=(50, 50),
        num_agents=50,
        agent_view_size=5,
        step_limit=5000,
    )

    for i in range(50, 100):
        env.num_agents = i
        main(
            model_path=model_path,
            env=env,
            model_class=CNN1,
        )

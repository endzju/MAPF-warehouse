import time

from src.agents.action_agent import ActionAgent
from src.core.MultiRobotGridEnv import MultiRobotGridEnv
from src.neural_networks.DQN.dqn import DQNet1, DQNet2, DQNet3
from src.utils.model_loader import load_model


def main(
    model_path="DQN_model_5.pth",
    env: MultiRobotGridEnv = None,
    model_class: DQNet1 | DQNet2 | DQNet3 = DQNet1,
):

    if ".pth" not in model_path:
        model_path += ".pth"
    # view_size = int(model_path.split("_")[-1].split(".")[0])

    observations, info = env.reset()

    terminated = False
    truncated = False
    total_step = 0
    # id_to_agent = {agent.id: agent for agent in env.agents}

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

        time.sleep(0.4)

    print("Symulacja zakończona")


if __name__ == "__main__":
    model_path = "DQNet1_3_5.pth"
    env = MultiRobotGridEnv(
        grid_size=(10, 10),
        num_agents=1,
        agent_view_size=5,
        step_limit=5000,
    )

    for i in range(1, 20):
        env.num_agents = i
        main(
            model_path=model_path,
            env=env,
            model_class=DQNet1,
        )

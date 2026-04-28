import time

from src.core.MultiRobotGridEnv import MultiRobotGridEnv
from src.neural_networks.DQN.dqn import DQNAgent
from src.utils.model_loader import load_model


def main():
    # 1. Inicjalizacja środowiska
    env = MultiRobotGridEnv(
        grid_size=(5, 5), num_agents=1, agent_view_size=5, step_limit=5000
    )

    # 2. Reset środowiska - otrzymujemy początkowe obserwacje
    observations, info = env.reset()

    terminated = False
    truncated = False
    total_step = 0
    id_to_agent = {agent.id: agent for agent in env.agents}

    # 3. Główna pętla symulacji
    while not (terminated or truncated):
        # Generujemy akcje dla każdego agenta, który jest jeszcze w grze
        # W Twoim środowisku akcje to: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT, 4=WAIT
        model = load_model("DQN_model_5.pth")
        model.eval()
        dqn_agent = DQNAgent(model, epsilon=0, epsilon_min=0, decay=0)
        # actions = {agent_id: env.np_random.integers(5) for agent_id in observations}
        actions = {
            agent_id: dqn_agent.get_action(obs, device="cpu")
            for agent_id, obs in observations.items()
        }
        # Wykonujemy krok w środowisku
        observations, rewards, terminated, truncated, info = env.step(actions)

        total_step += 1

        # Renderowanie i debugowanie
        print(f"--- Krok: {total_step} ---")
        print(f"Nagrody: {rewards}")
        env.render()

        # Opcjonalne spowolnienie, żeby dało się coś zauważyć w konsoli
        time.sleep(0.1)

        if terminated:
            print("Gratulacje! Zadanie zostało wykonane.")

    print("Symulacja zakończona!")


if __name__ == "__main__":
    main()

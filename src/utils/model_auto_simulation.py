import time
from pathlib import Path

from stable_baselines3 import A2C, PPO

from src.core.MultiRobotGridEnv import MultiRobotGridEnv
from src.models.depot import Depot


def simulate(models: list, env: MultiRobotGridEnv, render=False, iters=10) -> float:
    paused = False
    sum_time = 0
    for i in range(iters):
        obs, info = env.reset()
        if render:
            env.render()
        done = False
        while not done:
            if render:
                quit_requested, pause_pressed = env.handle_events()

                if pause_pressed:
                    paused = not paused

                if quit_requested:
                    break

            if render and paused:
                env.render(paused=True)
                time.sleep(0.1)
                continue

            actions1, _states = models[0].predict(obs, deterministic=True)
            actions2, _states = models[1].predict(obs, deterministic=True)

            final_actions = []

            for i in range(len(actions1)):
                if i % 2 == 0:
                    final_actions.append(actions1[i])
                else:
                    final_actions.append(actions2[i])

            obs, reward, terminated, truncated, info = env.step(final_actions)
            done = terminated or truncated
            if render:
                env.render()
                time.sleep(0.3)
        sum_time += env.step_count

    return sum_time / iters


data_dir = Path(__file__).parent.parent / "data" / "comp_intel"
data_dir.mkdir(exist_ok=True, parents=True)


depot1 = Depot((0, 0))
depot2 = Depot((5, 5))

env = MultiRobotGridEnv(
    grid_size=(6, 6),
    agent_view_size=3,
    input_depots=[depot1],
    output_depots=[depot2],
    step_limit=5000,
    task_length=5,
    max_robots=5,
    num_tasks=10,
)

model_ppo_a_path = data_dir / "PPO_a_mapf_warehouse"
model_ppo_b_path = data_dir / "PPO_b_mapf_warehouse"
model_a2c_path = data_dir / "A2C_mapf_warehouse"
model_ppo_a = PPO.load(str(model_ppo_a_path))
model_ppo_b = PPO.load(str(model_ppo_b_path))
model_a2c = A2C.load(str(model_a2c_path))

simulation_time = simulate([model_ppo_a, model_ppo_a], env, render=False, iters=10)
print(f"Symulacja PPO_a: {simulation_time} steps")
simulation_time = simulate([model_ppo_b, model_ppo_b], env, render=False, iters=10)
print(f"Symulacja PPO_b: {simulation_time} steps")
simulation_time = simulate([model_a2c, model_a2c], env, render=False, iters=10)
print(f"Symulacja A2C: {simulation_time} steps")
simulation_time = simulate([model_ppo_a, model_a2c], env, render=False, iters=10)
print(f"Symulacja PPO i A2C: {simulation_time} steps")


print("Symulacja zakończona.")

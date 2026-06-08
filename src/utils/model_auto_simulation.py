import time
from pathlib import Path

from stable_baselines3 import PPO

from src.core.MultiRobotGridEnv import MultiRobotGridEnv
from src.models.depot import Depot

data_dir = Path(__file__).parent.parent / "data" / "comp_intel"
data_dir.mkdir(exist_ok=True, parents=True)
model_path = data_dir / "ppo_mapf_warehouse"

depot1 = Depot((0, 0))
depot2 = Depot((5, 5))

env = MultiRobotGridEnv(
    grid_size=(6, 6),
    agent_view_size=3,
    input_depots=[depot1],
    output_depots=[depot2],
    step_limit=100000,
    task_length=5,
    max_robots=5,
    num_tasks=10,
)

model = PPO.load(str(model_path))

obs, info = env.reset()
env.render()
done = False

print("Rozpoczynam symulację...")
paused = False
while not done:
    quit_requested, pause_pressed = env.handle_events()

    if pause_pressed:
        paused = not paused

    if quit_requested:
        break

    if paused:
        env.render(paused=True)
        time.sleep(0.1)
        continue

    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    done = terminated or truncated

    time.sleep(0.3)

print("Symulacja zakończona.")

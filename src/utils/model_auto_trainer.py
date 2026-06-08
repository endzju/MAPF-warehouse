from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import A2C, PPO  # noqa: F401
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy

from src.core.MultiRobotGridEnv import MultiRobotGridEnv
from src.models.depot import Depot

depot1 = Depot((0, 0))
depot2 = Depot((5, 5))

data_dir = Path(__file__).parent.parent / "data" / "comp_intel"
data_dir.mkdir(exist_ok=True, parents=True)

env = MultiRobotGridEnv(
    grid_size=(6, 6),
    agent_view_size=3,
    input_depots=[depot1],
    output_depots=[depot2],
    step_limit=200,
    task_length=5,
    max_robots=5,
    num_tasks=10,
)
env = Monitor(env, str(data_dir))

# algorithm = PPO
algorithm = A2C

model_path = data_dir / f"{algorithm.__name__}_mapf_warehouse"

model = algorithm("MultiInputPolicy", env, verbose=1, tensorboard_log=str(data_dir))

total_timesteps = 500_000

model.learn(total_timesteps=total_timesteps)

model.save(str(model_path))


x, y = ts2xy(load_results(data_dir), "timesteps")

window = 50
if len(y) > window:
    y_smooth = np.convolve(y, np.ones(window) / window, mode="valid")
    x_smooth = x[window - 1 :]
else:
    y_smooth = y
    x_smooth = x

plt.figure(figsize=(10, 5))
plt.plot(x_smooth, y_smooth, label="Mean reward")
plt.title(f"{algorithm.__name__} learning curve")
plt.xlabel("Timesteps")
plt.ylabel("Reward")
plt.grid(True)
plt.legend()

plt.savefig(str(data_dir / f"learning_curve_{algorithm.__name__}.png"), dpi=300)
print("Wykres został zapisany jako 'learning_curve.png'")

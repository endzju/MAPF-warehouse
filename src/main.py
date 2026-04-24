from src.core.MultiRobotGridEnv import MultiRobotGridEnv

if __name__ == "__main__":
    env = MultiRobotGridEnv()
    env.reset()

    print(env.render())

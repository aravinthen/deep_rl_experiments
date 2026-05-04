import gymnasium as gym
import time

env = gym.make('CartPole-v1', render_mode="human")

# reset environment to start a new episode
observation, info = env.reset()

print(f"Starting obs: {observation}")

done = False
total_reward = 0

while not done:
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    time.sleep(0.1)

    total_reward += reward
    done = terminated or truncated

print(f"Episode finished with total reward {total_reward}")
env.close()
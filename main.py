"""
DQN agent for the continuous maze environment (env.py).

How to run (from inside this folder):
    Test (default):  test_dqn=True, train_dqn=False, render=True, save_video=True  -> loads dqn_1.pth, runs greedy,
                     opens the window, saves solution.mp4
    Train:           train_dqn=True, test_dqn=False   -> saves dqn_1.pth + training_curve.png

env.py is unmodified.
"""

# Imports:
# --------
import os
import torch
import pygame
import numpy as np
from DQN_model import Qnet
import torch.optim as optim
import matplotlib.pyplot as plt
from utils import ReplayBuffer, train
from env import ContinuousMazeEnv


# What to run:
train_dqn = False
test_dqn = True
render = True
save_video = False

dim_actions = 4   # up, down, left, right
dim_states = 2    # position (x, y)
exp_no = 1

# Hyperparameters:
learning_rate = 0.005
gamma = 0.98
buffer_limit = 50_000
batch_size = 32
num_episodes = 10_000
max_steps = 300
shaping_scale = 10

# Paths anchored to this file's folder so it runs from any working directory:
save_dir   = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(save_dir, f"dqn_{exp_no}.pth")
curve_path = os.path.join(save_dir, "training_curve.png")
video_path = os.path.join(save_dir, "solution.mp4")


if train_dqn:
    env = ContinuousMazeEnv()
    goal = env.goal_pos

    q_net = Qnet(dim_actions=dim_actions, dim_states=dim_states)
    q_target = Qnet(dim_actions=dim_actions, dim_states=dim_states)
    q_target.load_state_dict(q_net.state_dict())

    memory = ReplayBuffer(buffer_limit=buffer_limit)
    optimizer = optim.Adam(q_net.parameters(), lr=learning_rate)

    print_interval = 20
    episode_reward = 0.0
    rewards = []
    goals = []   # did the agent reach the goal each episode?

    for n_epi in range(num_episodes):
        # Start fully random, anneal epsilon to 1% over the first half of training.
        epsilon = max(0.01, 1.0 - n_epi / (num_episodes * 0.5))

        s, _ = env.reset()
        done = False
        reached_goal = False

        for _ in range(max_steps):
            a = q_net.sample_action(torch.from_numpy(s).float(), epsilon)
            s_prime, r, done, _, _ = env.step(a)

            # True reward for the curve / goal check (kept separate from shaping):
            episode_reward += r
            if done and r > 0:
                reached_goal = True

            # Reward shaping: reward getting closer to the goal (training only).
            shaped_r = r + shaping_scale * (np.linalg.norm(s - goal) - np.linalg.norm(s_prime - goal))

            done_mask = 0.0 if done else 1.0
            memory.put((s, a, shaped_r, s_prime, done_mask))
            s = s_prime

            if done:
                break

        if memory.size() > 2_000:
            train(q_net, q_target, memory, optimizer, batch_size, gamma)

        # Sync the target net and print progress every print_interval episodes.
        if n_epi % print_interval == 0 and n_epi != 0:
            q_target.load_state_dict(q_net.state_dict())
            print(f"n_episode :{n_epi}, Episode reward : {episode_reward}, n_buffer : {memory.size()}, eps : {epsilon}")

        rewards.append(episode_reward)
        goals.append(reached_goal)
        episode_reward = 0.0

        # Stop once the goal is reached in 20 consecutive episodes.
        if len(goals) >= 20 and all(goals[-20:]):
            print(f"Solved! Goal reached consistently by episode {n_epi}")
            break

    env.close()

    torch.save(q_net.state_dict(), model_path)

    plt.plot(rewards, label='Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.legend()
    plt.savefig(curve_path)
    plt.show()


if test_dqn:
    print("Testing the trained DQN: ")
    env = ContinuousMazeEnv()

    dqn = Qnet(dim_actions=dim_actions, dim_states=dim_states)
    dqn.load_state_dict(torch.load(model_path))

    frames = []

    for _ in range(10):
        s, _ = env.reset()
        episode_reward = 0

        for _ in range(max_steps):
            action = dqn(torch.from_numpy(s).float())   # greedy, no exploration
            s_prime, reward, done, _, _ = env.step(action.argmax().item())
            s = s_prime
            episode_reward += reward

            if render or save_video:
                env.render()
                if save_video:
                    frame = pygame.surfarray.array3d(env.screen).transpose(1, 0, 2)
                    frames.append(frame)

            if done:
                break
        print(f"Episode reward: {episode_reward}")

    env.close()

    if save_video and frames:
        import imageio
        imageio.mimsave(video_path, frames, fps=15, macro_block_size=1)
        print(f"Saved solution.mp4  ({len(frames)} frames)")

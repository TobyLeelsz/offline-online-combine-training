import gym
import pybullet_envs
import numpy as np
from collections import deque
import torch
import wandb
import argparse
from buffer import ReplayBuffer
import glob
from utils import save, collect_random
import random
import pickle
from agent import SAC


def get_config():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument("--run_name", type=str, default="SAC",)
    parser.add_argument("--env", type=str, default="LunarLander-v2")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--buffer_size", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--log_video", type=int, default=0)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--n_episode", type=int, default=3000)
    parser.add_argument("--n_steps_max", type=int, default=1000)
    args = parser.parse_args()
    return args


def generate_dataset(config):
    env = gym.make(config.env)
    env.seed(config.seed)
    env.action_space.seed(config.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    agent = SAC(state_size=env.observation_space.shape[0],
                         action_size=env.action_space.n,
                         device=device)
    actor_state_dict = torch.load("trained_models/SACSAC_discrete0.pth", map_location=device)
    agent.actor_local.load_state_dict(actor_state_dict)
    n_episode = config.n_episode

    paths = []
    max_step = config.n_steps_max

    for i in range(n_episode):
        state = env.reset()
        print("Initial State:", state)
        done = False
        episode_return = 0
        state_list = []
        action_list = []
        reward_list = []
        next_state_list = []
        done_list = []
        step = 0
        while not done and step < max_step:
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            state_list.append(state)
            action_list.append(action)
            reward_list.append(reward)
            next_state_list.append(next_state)
            done_list.append(done)
            episode_return += reward
            state = next_state
            step += 1

        state_list = np.array(state_list)
        action_list = np.array(action_list)
        reward_list = np.array(reward_list)
        next_state_list = np.array(next_state_list)
        done_list = np.array(done_list)

        paths.append([state_list, action_list, reward_list, next_state_list, done_list])
        
        print(f"Episode {i} Return: {episode_return}")

    with open("offline_dataset.dat", "wb") as f:
        pickle.dump(paths, f)







if __name__ == "__main__":
    config = get_config()
    generate_dataset(config)
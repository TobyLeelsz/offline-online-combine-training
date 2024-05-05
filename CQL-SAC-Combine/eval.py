import gym
import numpy as np
from collections import deque
import wandb
import torch
import argparse
import pickle
from agent import CQLSAC


def get_config():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument("--run_name", type=str, default="CQL-SAC-eval",)
    parser.add_argument("--env", type=str, default="LunarLander-v2")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--buffer_size", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--log_video", type=int, default=0)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--n_episode", type=int, default=1000)
    args = parser.parse_args()
    return args


def evaluate(config):
    env = gym.make(config.env)
    average100 = deque(maxlen=100)
    env.seed(config.seed)
    env.action_space.seed(config.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    agent = CQLSAC(state_size=env.observation_space.shape[0],
                         action_size=env.action_space.n,
                         device=device)
    actor_state_dict = torch.load(f"trained_models/CQL-SAC-discreteCQL-SAC-discrete{config.episodes}.pth", map_location=device)
    agent.actor_local.load_state_dict(actor_state_dict)
    n_episode = config.n_episode

    paths = []

    with wandb.init(project="CQL-SAC-eval", name=config.run_name, config=config):

        for i in range(n_episode):
            state = env.reset()
            print("Initial State:", state)
            done = False
            rewards = 0
            state_list = []
            action_list = []
            reward_list = []
            next_state_list = []
            done_list = []
            steps = 0
            while not done:
                action = agent.get_action(state)
                next_state, reward, done, _ = env.step(action)
                next_state_list.append(next_state)
                done_list.append(done)
                rewards += reward
                state = next_state
                steps += 1

            state_list = np.array(state_list)
            action_list = np.array(action_list)
            reward_list = np.array(reward_list)
            next_state_list = np.array(next_state_list)
            done_list = np.array(done_list)

            paths.append([state_list, action_list, reward_list, next_state_list, done_list])
            average100.append(rewards)

            print(f"Episode {i} Return: {rewards}")

            wandb.log({"Reward": rewards,
                        "Steps": steps,
                        "Episode": i,
                        "Average100": np.mean(average100),
                        })

        with open("datasets\\test_episodes.dat", "wb") as f:
            pickle.dump(paths, f)

if __name__ == "__main__":
    config = get_config()
    evaluate(config)
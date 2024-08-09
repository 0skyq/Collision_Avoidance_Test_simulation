import os
import sys
import numpy as np
import pickle
import torch
import random
import time
from datetime import datetime
from encoder_init import EncodeState
from networks.on_policy.ppo.agent import PPOAgent
from simulation.connection import ClientConnection
from simulation.environment import CarlaEnvironment
from parameters import *

def runner():
    exp_name = 'ppo'
    train = False
    town = "Town07"
    checkpoint_load = True
    test_timesteps = 10000
    action_std_init = 0.5

    if exp_name == 'ppo':
        run_name = "PPO"
    else:
        sys.exit("Unsupported experiment name")

    # Seeding
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True

    timestep = 0
    episode = 0
    cumulative_score = 0
    episodic_length = []
    scores = []
    deviation_from_center = 0
    distance_covered = 0

    try:
        client, world = ClientConnection(town).setup()
        print("Connection has been setup successfully.")
    except Exception as e:
        print(f"Connection failed: {e}")
        sys.exit()

    env = CarlaEnvironment(client, world, town)
    encode = EncodeState(LATENT_DIM)

    try:
        time.sleep(0.5)

        if checkpoint_load:
            chkt_file_nums = len(next(os.walk(f'checkpoints/PPO/{town}'))[2]) - 1
            chkpt_file = f'checkpoints/PPO/{town}/checkpoint_ppo_'+str(chkt_file_nums)+'.pickle'
            with open(chkpt_file, 'rb') as f:
                data = pickle.load(f)
                episode = data['episode']
                timestep = data['timestep']
                cumulative_score = data['cumulative_score']
                action_std_init = data['action_std_init']
            agent = PPOAgent(town, action_std_init)
            agent.load()
        else:
            sys.exit("No checkpoint found for testing.")

        while timestep < test_timesteps:
            observation = env.reset()
            observation = encode.process(observation)

            current_ep_reward = 0
            t1 = datetime.now()

            for t in range(EPISODE_LENGTH):
                action = agent.get_action(observation, train=False)
                observation, reward, done, info = env.step(action)
                if observation is None:
                    break
                observation = encode.process(observation)
                
                timestep += 1
                current_ep_reward += reward
                if done:
                    episode += 1
                    t2 = datetime.now()
                    t3 = t2 - t1
                    episodic_length.append(abs(t3.total_seconds()))
                    break

            deviation_from_center += info[1]
            distance_covered += info[0]
            scores.append(current_ep_reward)
            cumulative_score = np.mean(scores)

            print(f'Episode: {episode}, Timestep: {timestep}, Reward:  {current_ep_reward:.2f}, Average Reward:  {cumulative_score:.2f}')

            # Log metrics here as needed

            episodic_length = []
            deviation_from_center = 0
            distance_covered = 0

        print("Terminating the run.")
    finally:
        sys.exit()

if __name__ == "__main__":
    try:        
        runner()
    except KeyboardInterrupt:
        sys.exit()
    finally:
        print('\nExit')

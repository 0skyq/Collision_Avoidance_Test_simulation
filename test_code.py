import os
import sys
import time
import random
import numpy as np
import pickle
import torch
from datetime import datetime
from encoder_init import EncodeState
from networks.on_policy.ppo.agent import PPOAgent
from simulation.connection import ClientConnection
from simulation.environment import CarlaEnvironment
from parameters import *

def runner():
    train = True
    town = "Town02"
    #checkpoint_load = True
    total_timesteps = 500
    action_std_init = 0.5

    # Seeding
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True

    action_std_decay_rate = 0.05
    min_action_std = 0.05   
    action_std_decay_freq = 5e5
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

    print(f"EPISODE_LENGTH = {EPISODE_LENGTH} LATENT_DIM = {LATENT_DIM}")

    try:
        time.sleep(0.5)
        agent = PPOAgent(town, action_std_init)

        while timestep < total_timesteps:
            print("training...")
            observation = env.reset()
            observation = encode.process(observation)
            print("Reading the Environment...")

            current_ep_reward = 0
            t1 = datetime.now()

            for t in range(EPISODE_LENGTH):
                #print("In the episode loop")
                action = agent.get_action(observation, train=True)
                #print(action)

                observation, reward, done, info = env.step(action)
                #print(observation)

                if observation is None:
                    print("There is no observation")
                    break

                observation = encode.process(observation)
                
                agent.memory.rewards.append(reward)
                agent.memory.dones.append(done)
                
                timestep += 1
                current_ep_reward += reward
                
                if timestep % action_std_decay_freq == 0:
                    action_std_init = agent.decay_action_std(action_std_decay_rate, min_action_std)

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

        print("Terminating the run.")

    finally:
        try:
            print("Saving final checkpoint...")
            agent.save()
        except Exception as e:
            print(f"An error occurred while saving checkpoint: {e}")
        sys.exit()

if __name__ == "__main__":
    try:        
        runner()
    except KeyboardInterrupt:
        sys.exit()
    finally:
        print('\nExit')

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
    exp_name = 'ppo'
    train = True
    town = "Town02"
    checkpoint_load = True
    total_timesteps = 1000000
    action_std_init = 0.5

    try:
        if exp_name == 'ppo':
            run_name = "PPO"
        else:
            """
            
            Here the functionality can be extended to different algorithms.

            """ 
            sys.exit() 
    except Exception as e:
        print(e.message)
        sys.exit()

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

        if checkpoint_load:

            if len(next(os.walk(f'checkpoints/PPO/{town}'))[2]) == 0:
                print(f"number of check points : {len(next(os.walk(f'checkpoints/PPO/{town}'))[2])}")
                chkt_file_nums = -1

            else:
                print(f"number of check points : {len(next(os.walk(f'checkpoints/PPO/{town}'))[2])}")
                print("Check_point loaded")
                chkt_file_nums = len(next(os.walk(f'checkpoints/PPO/{town}'))[2]) - 1
                chkpt_file = f'checkpoints/PPO/{town}/checkpoint_ppo_'+str(chkt_file_nums)+'.pickle'

                print(chkpt_file)

                with open(chkpt_file, 'rb') as f:
                    data = pickle.load(f)
                    #print(data)
                    episode = data['episode']
                    timestep = data['timestep']
                    cumulative_score = data['cumulative_score']
                    action_std_init = data['action_std_init']

                agent = PPOAgent(town, action_std_init)
                agent.load()


        if checkpoint_load == False or chkt_file_nums == -1:  
            agent = PPOAgent(town, action_std_init)

        while timestep < total_timesteps:
            print("training...")
            observation = env.reset()
            observation = encode.process(observation)

            current_ep_reward = 0
            t1 = datetime.now()

            for t in range(EPISODE_LENGTH):
                action = agent.get_action(observation, train=True)
                observation, reward, done, info = env.step(action)
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

                if timestep == total_timesteps - 1:
                    agent.chkpt_save()

                if done:
                    episode += 1
                    t2 = datetime.now()
                    t3 = t2 - t1
                    episodic_length.append(abs(t3.total_seconds()))
                    break
            
            deviation_from_center += info[1]
            distance_covered += info[0]
            scores.append(current_ep_reward)
            cumulative_score = np.mean(scores) if not checkpoint_load else ((cumulative_score * (episode - 1)) + current_ep_reward) / episode

            print(f'Episode: {episode}, Timestep: {timestep}, Reward:  {current_ep_reward:.2f}, Average Reward:  {cumulative_score:.2f}')

            if episode % 10 == 0:

                checkpoint_dir = f'checkpoints/PPO/{town}'
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)

                print("episode_%10")
                agent.learn()
                agent.chkpt_save()
                chkt_file_nums = len(next(os.walk(f'checkpoints/PPO/{town}'))[2])
                chkpt_file = f'checkpoints/PPO/{town}/checkpoint_ppo_'+str(chkt_file_nums-1)+'.pickle'
                data_obj = {'cumulative_score': cumulative_score, 'episode': episode, 'timestep': timestep, 'action_std_init': action_std_init}
                with open(chkpt_file, 'wb') as handle:
                    pickle.dump(data_obj, handle)
            
            if episode % 5 == 0:
                print("episode_%5")
                episodic_length = []
                deviation_from_center = 0
                distance_covered = 0

            if episode % 100 == 0:
                print("episode_%100")
                agent.save()
                chkt_file_nums = len(next(os.walk(f'checkpoints/PPO/{town}'))[2])
                chkpt_file = f'checkpoints/PPO/{town}/checkpoint_ppo_'+str(chkt_file_nums-1)+'.pickle'
                data_obj = {'cumulative_score': cumulative_score, 'episode': episode, 'timestep': timestep, 'action_std_init': action_std_init}
                with open(chkpt_file, 'wb') as handle:
                    pickle.dump(data_obj, handle)

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

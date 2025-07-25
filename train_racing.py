import os
import time
from datetime import datetime
import argparse
import gymnasium as gym
import numpy as np
import torch
from ppo import PPO

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.GateAviary import GateAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
num_gates = [0, 0, 0, 0, 0, 0]

def train():
    DEFAULT_OBS = ObservationType('kin')
    DEFAULT_ACT = ActionType('rpm')
    DEFAULT_GUI = True
    DEFAULT_RECORD_VIDEO = False
    DEFAULT_OUTPUT_FOLDER = 'results'
    DEFAULT_COLAB = False


    env = GateAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT)
    # init agent
    state_dim = 36
    action_dim = 4
    action_std = 0.6                    # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)
    #####################################################

    ################ PPO hyperparameters ################
    #update_timestep = max_ep_len * 4      # update policy every n timesteps
    max_training_timesteps = int(3e6)   # break training loop if timeteps > max_training_timesteps
    K_epochs = 80               # update policy for K epochs in one PPO update

    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor
    lr_actor = 0.0003       # learning rate for actor network
    lr_critic = 0.001       # learning rate for critic network

    random_seed = 0         # set random seed if required (0 = no random seed)
    log_dir = "log_dir/"
    run_num = "racing/"
    log_f_name = log_dir + run_num + 'PPO_log' + ".csv"
    if not os.path.exists(os.path.join(log_dir, str(run_num))):
        os.mkdir(os.path.join(log_dir, str(run_num)))
    checkpoint_path = log_dir + "ppo_drone.pth"

    print("current logging run number for " + " gym pybulet drone : ", run_num)
    print("logging at : " + log_f_name)
    log_f = open(log_f_name,"w+")
    log_f.write('episode,timestep,reward\n')
    update_timestep = env.EPISODE_LEN_SEC*env.CTRL_FREQ * 4
    print_freq = env.EPISODE_LEN_SEC*env.CTRL_FREQ  * 10        # print avg reward in the interval (in num timesteps)
    log_freq =  env.EPISODE_LEN_SEC*env.CTRL_FREQ
    save_model_freq = int(1e5)          # save model frequency (in num timesteps)
    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std)

    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    time_step = 0
    i_episode = 0
    best_reward = 0.0
    while time_step <= max_training_timesteps:
        obs, info = env.reset(seed=42, options={})

        current_ep_reward = 0
        for i in range(env.EPISODE_LEN_SEC*env.CTRL_FREQ):
            action = ppo_agent.select_action(obs)
            action = np.expand_dims(action, axis=0)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step += 1
            current_ep_reward += reward

            # update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent.update()

            if time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

            if time_step % log_freq == 0:

                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)

                log_f.write('{}, {}, {}\n'.format(i_episode, time_step, log_avg_reward))
                log_f.flush()

                log_running_reward = 0
                log_running_episodes = 0

            if time_step % print_freq == 0:

                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))


                print_running_reward = 0
                print_running_episodes = 0

            if current_ep_reward > best_reward:
                best_reward = current_ep_reward
                ppo_agent.save(os.path.join(log_dir, str(run_num), f'best_model_{best_reward:.1f}.pth'))
            elif env.passing_flag[5]:
                ppo_agent.save(os.path.join(log_dir, str(run_num), f'wow_model_{best_reward:.1f}.pth'))

            # break; if the episode is over
            if done or i == (env.EPISODE_LEN_SEC)*env.CTRL_FREQ - 1:
                num_gates[sum(env.passing_flag)] += 1
            if done:
                break


        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1
        print(num_gates)


    log_f.close()
    env.close()



    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


if __name__ == '__main__':
    train()
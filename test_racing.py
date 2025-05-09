import os
import time
from datetime import datetime
import argparse
import gymnasium as gym
import numpy as np
import torch
from ppo import PPO
import matplotlib.pyplot as plt

from gym_pybullet_drones.envs.GateAviary import GateAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType


#################################### Testing ###################################
def test():
    print("============================================================================================")

    ################## hyperparameters ##################

    max_ep_len = 1800           # max timesteps in one episode
    action_std = 0.1           # set same std for action distribution which was used while saving

    render = True              # render environment on screen
    frame_delay = 0             # if required; add delay b/w frames

    total_test_episodes = 3    # total num of testing episodes

    K_epochs = 80               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor

    lr_actor = 0.0003           # learning rate for actor
    lr_critic = 0.001           # learning rate for critic

    #####################################################
    DEFAULT_OBS = ObservationType('kin')
    DEFAULT_ACT = ActionType('rpm')
    DEFAULT_GUI = True
    DEFAULT_RECORD_VIDEO = False
    DEFAULT_OUTPUT_FOLDER = 'results'
    filename = os.path.join(DEFAULT_OUTPUT_FOLDER, 'recording_'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
    if not os.path.exists(filename):
        print(filename)
        os.makedirs(filename+'/')

    env = GateAviary(gui=DEFAULT_GUI,
                           obs=DEFAULT_OBS,
                           act=DEFAULT_ACT,
                           record=DEFAULT_RECORD_VIDEO)

    state_dim = 36
    action_dim = 4

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std)
    checkpoint_path = "log_dir/racing/wow_model_70.7.pth"
    print("loading network from : " + checkpoint_path)

    ppo_agent.load(checkpoint_path)

    print("--------------------------------------------------------------------------------------------")

    test_running_reward = 0

    for ep in range(1, total_test_episodes+1):
        ep_reward = 0
        state = env.reset()

        obs, info = env.reset(seed=42, options={})
        ep_reward = 0
        start_time = datetime.now().replace(microsecond=0)
        start = time.time()
        for i in range((env.EPISODE_LEN_SEC+10)*env.CTRL_FREQ):
            action = ppo_agent.select_action(obs)
            action = np.expand_dims(action, axis=0)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            env.render()
            sync(i, start, env.CTRL_TIMESTEP)
            if truncated or terminated:
                break

        test_running_reward +=  ep_reward
        print('Episode: {} \t\t Reward: {}'.format(0, round(ep_reward, 2)))

    # clear buffer
    ppo_agent.buffer.clear()

    env.close()

    print("============================================================================================")

    avg_test_reward = test_running_reward / total_test_episodes
    avg_test_reward = round(avg_test_reward, 2)
    print("average test reward : " + str(avg_test_reward))

    print("============================================================================================")


if __name__ == '__main__':
    test()

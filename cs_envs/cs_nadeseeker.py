import hashlib
import os

import gymnasium as gym
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
from rich import print

from .cs_nadeseeker_core import CSNadeSeekerCore


class CSNadeSeekerEnv(gym.Env):
    def __init__(self, obj_pos, initial_ppos, initial_ang_bias, 
                 cs2_path, mov_size=1, episode_length=50, 
                 print_info=True, map_name='cs2', coef_flt_compensate=2):
        super().__init__()
        self.obj_pos = obj_pos
        self.initial_ppos = initial_ppos
        self.initial_ang_bias = initial_ang_bias
        self.cs2_path = cs2_path
        self.mov_size = mov_size
        self.episode_length = episode_length
        self.print_info = print_info
        self.map_name = map_name
        identify_info = self.map_name + '_' + self.obj_pos + '&' + self.initial_ppos
        self.identify_info = identify_info
        self.hash = hashlib.sha256(self.identify_info.encode()).digest()[:10].hex()
        self.coef_flt_compensate = coef_flt_compensate
                
        self.initialize_environment()

        return

    def initialize_environment(self):
        self.OUTPUT_DIR = './train'
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        self.initialize_action_observation_space()

        self.df_info_cols = [
            'ppos_x', 'ppos_y', 'ppos_z', 'pang_x', 'pang_y', 'pang_z', 
            'npos_x', 'npos_y', 'npos_z', 'flt', # flt means flying time
            'd_npos_x', 'd_npos_y', 'd_npos_z', 'd2obj', 
            'reward', 'is_terminated'
            ]
        self.state_cols = ['pang_x', 'pang_y', 'd_npos_x', 'd_npos_y', 'd_npos_z', 'flt']
        self.df_info = pd.DataFrame(columns=self.df_info_cols, dtype='Float32')
        self.df_info['is_terminated'] = self.df_info['is_terminated'].astype('bool')

        self.core = CSNadeSeekerCore(self.df_info, 
                                    self.obj_pos, self.initial_ppos, self.initial_ang_bias, 
                                    self.cs2_path, self.hash, self.coef_flt_compensate)
        return

    def initialize_action_observation_space(self):
        self.action_mapping_mov_dir = {
            0: ( 0, -1), 
            1: (-1,  0), 
            2: ( 1,  0), 
            3: ( 0,  1), 
            }
        self.action_space = gym.spaces.Discrete(len(self.action_mapping_mov_dir))
        
        # 'pang_x', 'pang_y', 'd_npos_x', 'd_npos_y', 'd_npos_z', 'flt 
        low  = np.array([-90, -180, -5000, -5000, -2000, -1], dtype='float32')
        high = np.array([ 90,  180,  5000,  5000,  2000, 12], dtype='float32')

        self.observation_space = gym.spaces.Box(low=low, high=high)
        return

    def step(self, action):

        # Update step
        self.current_step += 1
        print('Current step: ', self.current_step, end='; ')

        new_ang = self.mov_size * np.array(self.action_mapping_mov_dir[action])

        self.core.step_update(new_ang[0], new_ang[1])
        self.df_info = self.core.df_info
        new = self.df_info.iloc[self.current_step, :]

        # Obs
        self.state = new[self.state_cols].values.astype('float32')
        obs = self.state

        # Reward
        reward = new['reward']  

        # Terminated
        terminated = new['is_terminated']

        # Truncated
        truncated = self.current_step >= self.episode_length  # Truncate the episode after x steps

        info = new.to_dict()
        self.print_df_info()

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):

        if (self.core.is_session_initialized == False):
            self.core.session_init()
        else:
            self.core.reset_env()

        self.df_info = self.core.df_info
        self.current_step = 1

        # Reset the state of the environment to an initial state
        self.state = self.df_info.loc[self.current_step, self.state_cols].values.astype('float32')

        obs = self.state
        info = self.df_info.iloc[self.current_step, :].to_dict()

        self.print_df_info()
        return obs, info

    def render(self, mode='human'):
        # Not implemented
        return

    def print_df_info(self):
        if (self.print_info):
            print(self.df_info.tail(5).drop(columns=[
            'ppos_x', 'ppos_y', 'ppos_z', 
            'pang_z', 
            'npos_x', 'npos_y', 'npos_z'
            ]))
        return
    

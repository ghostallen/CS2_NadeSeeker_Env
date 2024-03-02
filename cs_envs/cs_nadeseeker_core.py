import os
import shutil
import threading
from time import sleep

import gymnasium as gym
import keyboard
import matplotlib
matplotlib.use('Agg') 
import numpy as np
import pandas as pd
import pyautogui as pg
from rich import print

import CS2PosParser
from .cs_nadeseeker_utils import CSNadeSeekerUtils


class CSNadeSeekerCore:
    def __init__(self, df_info: pd.DataFrame, 
                 obj_pos, initial_ppos, initial_ang_bias, 
                 cs2_path, hash, coef_flt_compensate, 
                 dist_dif_threshold=5, terminated_threshold=50):
        self.obj_pos = obj_pos
        self.initial_ppos = initial_ppos
        self.initial_ang_bias = initial_ang_bias
        
        self.OUTPUT_DIR = './agent_data'
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

        self.df_info = df_info
        self.cs2_path = cs2_path
        self.log_file = self.cs2_path + r"\game\csgo\console.log"
        self.tp_cfg = self.cs2_path + r"\game\csgo\cfg\tp.cfg"
        self.action_cfg = self.cs2_path + r"\game\csgo\cfg\action.cfg"
        self.hash = hash
        self.good_angles_path = './good_angles'
        os.makedirs(self.good_angles_path, exist_ok=True)
        self.good_angles_file = self.good_angles_path + '/' + self.hash + '.csv'

        self.coef_flt_compensate = coef_flt_compensate
        self.dist_dif_threshold = dist_dif_threshold
        self.terminated_threshold = terminated_threshold

        self.utils = CSNadeSeekerUtils(self.cs2_path)

        self.is_session_initialized = False

        self.npos_col = ['npos_x', 'npos_y', 'npos_z']
        self.ninfo_col = ['pang_x', 'pang_y', 'flt']
        self.ppos_col = ['ppos_x', 'ppos_y', 'ppos_z']

        self.ppos, self.npos = [], []

        self.good_angles = pd.DataFrame(columns=['pang_x', 'pang_y', 'flt'])

        return

    def session_init(self):
        self.utils.clear_tp_cfg()
        self.utils.str2tp_cfg(self.obj_pos)        
        # Go to 'initial_ppos'
        pg.press('num5')
        sleep(0.25)

        # Throw the grenade at foot; Initialize pandas datafarme
        print("Initializing...")
        self.utils.val2action_cfg(90, 0, 0)
        sleep(0.5)
        self.ppos, self.npos = self.get_pos()
        self.df_info.loc[0] = self.ppos + self.npos + \
            [0]*(self.df_info.shape[1] - len(self.ppos) - len(self.npos) - 1) + [False]
        self.df_info = self.df_info.rename(index={0: 'obj'})
        self.obj = self.df_info.iloc[0, :]

        # Backup 'initial_ppos'
        self.utils.str2tp_cfg(self.initial_ppos)
        shutil.copyfile(self.tp_cfg, self.tp_cfg + '.reset')

        self.reset_env()

        self.is_session_initialized = True
        return

    def reset_env(self):
        self.epi_filename = self.utils.create_unique_filename(self.OUTPUT_DIR)
        self.df_info.to_csv(os.path.join(self.OUTPUT_DIR, self.epi_filename), index=False)
        self.df_info = self.df_info.head(1).copy()

        self.utils.str2action_cfg(self.initial_ppos)

        shutil.copyfile(self.tp_cfg + '.reset', self.tp_cfg) # reset to initial tp info.
        sleep(0.5)
        pg.press('num5')
        sleep(0.25)

        ppos, npos = self.get_pos()
        self.last_action = np.array([ppos[-3], ppos[-2]])

        self.df_info.loc[len(self.df_info), :] = self.ppos + self.npos + \
            [0]*(self.df_info.shape[1] - len(self.ppos) - len(self.npos) - 1) + [False]

        print('Env is reset.')
        return
        
    def seek_nade(self, x: float, y: float):
        self.utils.val2action_cfg(x, y, 0.0)
        result = self.get_pos()
        return result

    def step_update(self, delta_x: float, delta_y: float):
        # Step cost
        reward = -1

        self.new_action = self.last_action + np.array([delta_x, delta_y])
        self.new_action[0] = self.utils.norm_ang_180(self.new_action[0])
        self.new_action[1] = self.utils.norm_ang_360(self.new_action[1])

        # Update state
        self.ppos, self.npos = self.seek_nade(self.new_action[0], self.new_action[1])
        self.last_action = self.new_action.copy()

        # Add the new data to df_info
        n_step = len(self.df_info)
        self.df_info.loc[n_step] = self.ppos + self.npos + \
            [0]*(self.df_info.shape[1] - len(self.ppos) - len(self.npos) - 1) + [False]
        
        new = self.df_info.iloc[-1, :]

        if (np.isnan(new['flt']) == True):
            # The grenade fly out of the map. Reward is 0.

            self.df_info.loc[n_step, ['flt', 'reward']] = -1, reward
            self.df_info.loc[n_step, :] = self.df_info.loc[n_step, :].fillna(self.df_info.mean())
            return

        self.df_info.loc[n_step, ['d_npos_x', 'd_npos_y', 'd_npos_z']] = [
            self.obj['npos_x'] - self.npos[0], 
            self.obj['npos_y'] - self.npos[1], 
            self.obj['npos_z'] - self.npos[2]
            ]
        self.new = self.df_info.iloc[-1, :]
        
        d_new2obj = self.utils.calc_dist(self.obj[self.npos_col].values, self.new[self.npos_col].values)
        self.df_info.loc[n_step, ['d2obj']] = d_new2obj


        pre, new = self.df_info.iloc[-2, :], self.df_info.iloc[-1, :]
        reward += 0.5 if (pre['d2obj'] - new['d2obj']) > self.dist_dif_threshold else 0

        terminated = True if new['d2obj'] < self.terminated_threshold else False

        self.df_info.loc[n_step, ['is_terminated']] = terminated
        if (terminated == True):
            reward += 100
            self.save_good_angle(self.new[['pang_x', 'pang_y', 'npos_x', 'npos_y', 'npos_z', 'flt']])

        self.df_info.loc[n_step, ['reward', 'is_terminated']] = reward, terminated

        return

    def save_good_angle(self, ninfo):
        if (len(ninfo) != 6):
            print('Error for ninfo. Saving failure')
            return

        # npos_x, npos_y, npos_z, flt
        if os.path.exists(self.good_angles_file):
            df_good_angles = pd.read_csv(self.good_angles_file)
        else:
            df_good_angles = pd.DataFrame(columns = ['pang_x', 'pang_y', 'npos_x', 'npos_y', 'npos_z', 'flt'])

        df_good_angles.loc[len(df_good_angles), :] = ninfo
        df_good_angles.to_csv(self.good_angles_file, index=False)

        return df_good_angles


    def get_pos(self):
        self.ppos, self.npos = [], []
        flag_ppos, flag_npos = threading.Event(), threading.Event()
        thread = threading.Thread(target=self.get_ppos, args=(flag_ppos,))
        thread.start()

        # exec action (move crosshair)
        pg.press('num1') 
        sleep(0.25)

        # Throw nade; Run "getpos" command.
        pg.press('num2') 
        while True:
            if flag_ppos.is_set():
                break
        sleep(0.25)

        thread = threading.Thread(target=self.get_npos, args=(flag_npos, ))
        thread.start()
        # Join spec for viewing npos
        pg.press('num3') 

        while True:
            if flag_npos.is_set():
                break

        # Join back to CT
        pg.press('num4') 
        sleep(0.25)

        # Teleport to previous ppos.
        pg.press('num5')
        return self.ppos, self.npos

    def get_ppos(self, flag_ppos):
        self.ppos = CS2PosParser.get_ppos(self.log_file, True, self.tp_cfg)
        flag_ppos.set()
        print(r"Got 'ppos'", end='; ')
        return

    def get_npos(self, flag_npos):
        self.npos = CS2PosParser.get_npos(host='127.0.0.1', port=23927)
        self.npos[-1] *= self.coef_flt_compensate
        flag_npos.set()
        print(r"Got 'npos'", end='\n')
        return

import datetime
import os
import shutil
from time import sleep

import keyboard
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import pyautogui as pg
from rich import print

import CS2PosParser

class CSNadeSeekerUtils:
    def __init__(self, cs2_path):
        self.cs2_path = cs2_path
        self.log_file = self.cs2_path + r"\game\csgo\console.log"
        self.tp_cfg = self.cs2_path + r"\game\csgo\cfg\tp.cfg"
        self.action_cfg = self.cs2_path + r"\game\csgo\cfg\action.cfg"
        return

    def val2action_cfg(self, x:'float', y:'float', z:'float'):
        with open(self.action_cfg, 'w+', encoding='utf-8') as f:
            f.write(f'setang {x} {y} {z}\n')
        return
    
    def str2tp_cfg(self, command):
        with open(self.tp_cfg, 'w+', encoding='utf-8') as f:
            f.write(command)
        return


    def str2action_cfg(self, command):
        with open(self.action_cfg, 'w+', encoding='utf-8') as f:
            f.write(command)
        return

    def create_unique_filename(self, output):
        now = datetime.datetime.now()
        base_filename = "{day:02d}_{month:02d}_{year}_ID".format(
            day=now.day, month=now.month, year=now.year)
        filenames = os.listdir(output)
        file_IDs = [int(file.split('_')[-1].split('.')[0]) for file in filenames
                    if file.startswith(base_filename)]
        new_ID = max(file_IDs) + 1 if file_IDs else 0
        return "{base}_{id}.csv".format(base=base_filename, id=new_ID)

    
    def clear_action_cfg(self):
        with open(self.action_cfg, 'w+', encoding='utf-8') as f:
            f.truncate(0)
        return

    def clear_tp_cfg(self):
        with open(self.tp_cfg, 'w+', encoding='utf-8') as f:
            f.truncate(0)
        return

    
    @staticmethod
    def angle_diff_180(angle1, angle2):
        diff = (angle1 - angle2 + 90) % 180 - 90
        return abs(diff)
    
    @staticmethod
    def angle_diff_360(angle1, angle2):
        # Normalize the difference within the range of [-180, 180]
        diff = (angle1 - angle2 + 180) % 360 - 180
        # Return the absolute value of the difference
        return abs(diff)
    
    @staticmethod
    def norm_ang_180(ang):
        return (ang + 90) % 180 - 90
    
    @staticmethod
    def norm_ang_360(ang):
        return (ang + 180) % 360 - 180
    

    def calc_dist(self, pos_A, pos_B, x_weight=1, y_weight=1, z_weight=0.25):
        weights = np.array([x_weight, y_weight, z_weight])
        weighted_vector = np.multiply(pos_A - pos_B, weights)
        return np.linalg.norm(weighted_vector)



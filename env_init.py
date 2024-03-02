import configparser
import hashlib

import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback



class EnvSetter:
    def __init__(self, config_ini:str):
        self.config_ini = config_ini
        self.config = self.load_config()
        self.identify_info = self.get_identify_info()
        self.output = self.get_output_path()
        self.env_id = 'CS-SeekNade-onsite-standthrow'
        self.register_env()
        return
        
    def setattr_id_info(self, model):
        model.identify_info = self.identify_info
        model.hash = self.get_hash(self.identify_info)
        return model

    def load_config(self):
        config = configparser.ConfigParser()
        config.read(self.config_ini)
        return config

    def get_identify_info(self):
        identify_info = self.config['DEFAULT']['map_name'] + '_' +\
                        self.config['DEFAULT']['obj_pos'] + '&' + self.config['DEFAULT']['initial_ppos']
        return identify_info

    def get_output_path(self):
        output = 'trained_agents/' + self.config['DEFAULT']['map_name'] + '_' +\
                 self.get_hash(self.identify_info)
        return output

    @staticmethod
    def get_hash(string):
        return hashlib.sha256(string.encode()).digest()[:10].hex()

    def register_env(self):
        gym.envs.registration.register(
            id=self.env_id,
            entry_point='cs_envs.cs_nadeseeker:CSNadeSeekerEnv',
            kwargs={
                'map_name': self.config['DEFAULT']['map_name'], 
                'obj_pos': self.config['DEFAULT']['obj_pos'], 
                'initial_ppos': self.config['DEFAULT']['initial_ppos'], 
                'initial_ang_bias': float(self.config['DEFAULT']['initial_ang_bias']), 
                'cs2_path': self.config['DEFAULT']['cs2_path'], 
                'mov_size': float(self.config['DEFAULT']['mov_size']), 
                'episode_length': float(self.config['DEFAULT']['episode_length']), 
            }
        )

####################


class SaveOnInterval(BaseCallback):
    def __init__(self, save_interval, save_path):
        super(SaveOnInterval, self).__init__()
        self.save_interval = save_interval
        self.save_path = save_path
        return

    def _on_step(self) -> bool:
        if self.n_calls % self.save_interval == 0:
            self.model.save(self.save_path + str(self.n_calls))
        return True
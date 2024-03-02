import os

import gymnasium as gym
from stable_baselines3 import DQN
import keyboard

from env_init import *

playgrounds_list = []
for f in os.listdir('./playgrounds'):
    if f.endswith('ini'):
        playgrounds_list.append('./playgrounds/' + f)


print("Press 0 to start training. ")
keyboard.wait("0")  # Wait for any key press
print("GO. ")

for config in playgrounds_list: 


    print(config)

    env_setter = EnvSetter(config)
    env = gym.make(env_setter.env_id)

    model = DQN(env=env, tensorboard_log=env_setter.output+"/log_tensorboard/", 
                
                policy = 'MlpPolicy', 
                learning_rate=0.0005, 
                learning_starts=50, 
                batch_size=64, 
                gradient_steps=1, 
            
                exploration_fraction=0.05,
                exploration_initial_eps=0.5,
                exploration_final_eps=0.1, 

                gamma=0.96, 

                policy_kwargs=dict(net_arch=[128, 128]), 
                train_freq=5, 
                seed=0
                )
   # Add hash
    model = env_setter.setattr_id_info(model)    


    model.learn(total_timesteps=500, log_interval=1, progress_bar=True, 
                callback=SaveOnInterval(100, env_setter.output+"/"))

    print('Evaluating the agent.')

    env.close()



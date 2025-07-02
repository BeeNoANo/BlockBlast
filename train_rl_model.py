# train_rl_model.py

import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from Enviroment.block_blast_env import BlockBlastEnv

# Create the RL environment
vec_env = make_vec_env(BlockBlastEnv, n_envs=4) # Use 4 parallel environments for faster training

# Define the model. PPO is a good starting point.
# We use MultiInputPolicy because our observation space is a dictionary.
model = PPO("MultiInputPolicy", vec_env, verbose=1, 
            tensorboard_log="./ppo_block_blast_tensorboard/",
            n_steps=2048, # Number of steps to run per environment per update
            batch_size=64,
            learning_rate=0.0003,
            gamma=0.99, # Discount factor
            
            # TĂNG ent_coef để khuyến khích khám phá.
            # Giá trị mặc định thường là 0.01. Hãy thử tăng lên 0.05 hoặc 0.1
            ent_coef=0.08) 

# Create a callback to save the model periodically
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./models/', name_prefix='block_blast_ppo_model')

# Start training! This will take a long time (hours or even days).
# The AI will learn by playing thousands of games.
print("Starting training...")
try:
    model.learn(total_timesteps=5_000_000, callback=checkpoint_callback)
except KeyboardInterrupt:
    print("Training interrupted. Saving model...")
    model.save("interrupted_ppo_block_blast_model")

print("Training finished. Saving final model.")
model.save("final_ppo_block_blast_model")
print("Model saved as final_ppo_block_blast_model.zip")
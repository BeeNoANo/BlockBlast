# train_rl_model.py

import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
# from stable_baselines3.common.results_reporter import make_results_reporter # <--- REMOVE OR COMMENT THIS LINE
import numpy as np

# Assuming BlockBlastEnv is correctly imported or defined
from Enviroment.block_blast_env import BlockBlastEnv # Or wherever your env is located

# Define a custom callback to handle logging of scores
class CustomTrackingCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_scores = []
        self.episode_rewards = []

    def _on_rollout_end(self) -> None:
        if 'infos' in self.locals:
            for info in self.locals['infos']:
                if 'episode' in info:
                    self.episode_rewards.append(info['episode']['r'])
                    # Check if 'score' is in the episode info
                    if 'score' in info['episode']: # Access 'score' from the 'episode' dict if it's there
                        self.episode_scores.append(info['episode']['score'])
                    elif 'score' in info: # Fallback: if 'score' is directly in info
                         self.episode_scores.append(info['score'])


        # Log metrics (example) - It's better to use self.logger.record for TensorBoard
        # You can also print to console for quick checks
        if self.num_timesteps > 0 and len(self.episode_scores) > 0:
            mean_score = np.mean(self.episode_scores[-self.model.n_steps:]) # Example for logging per rollout
            mean_reward = np.mean(self.episode_rewards[-self.model.n_steps:])

            # Log to TensorBoard
            self.logger.record('rollout/ep_rew_mean', mean_reward)
            self.logger.record('rollout/ep_score_mean', mean_score)
            self.logger.record('rollout/n_episodes', len(self.episode_scores)) # Track number of episodes

            # Optional: Print to console
            print(f"Timestep: {self.num_timesteps} - Mean Score: {mean_score:.2f}, Mean Reward: {mean_reward:.2f}")


    def _on_step(self) -> bool:
        return True


# The rest of your train_rl_model.py remains similar:
# Create the RL environment
vec_env = make_vec_env(BlockBlastEnv, n_envs=4)

# Define the model. PPO is a good starting point.
model = PPO("MultiInputPolicy", vec_env, verbose=1,
            tensorboard_log="./ppo_block_blast_tensorboard/",
            n_steps=2048,
            batch_size=64,
            learning_rate=0.0003,
            gamma=0.99,
            ent_coef=0.08)

# Create a callback to save the model periodically
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./models/', name_prefix='block_blast_ppo_model')

# Instantiate your custom callback
custom_callback = CustomTrackingCallback()

# Start training!
print("Starting training...")
try:
    # Pass both callbacks as a list
    model.learn(total_timesteps=5_000_000, callback=[checkpoint_callback, custom_callback])
except KeyboardInterrupt:
    print("Training interrupted.")

model.save("final_ppo_block_blast_model")
print("Training complete. Model saved.")
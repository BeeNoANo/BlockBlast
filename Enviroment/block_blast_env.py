# block_blast_env.py
import pygame
import sys
import gymnasium as gym
from gymnasium import spaces
import numpy as np
# Giả định cấu trúc thư mục là 'Helper/game_logic.py'
from Helper.game_logic import BlockBlastLogic, GRID_SIZE, SHAPES 

def pad_block(block):
    """
    Pads a block matrix to a fixed size of 5x5.
    This size is chosen to accommodate the largest block shape (e.g., 1x5 line).
    """
    padded = np.zeros((5, 5), dtype=np.float32)
    
    if not isinstance(block, np.ndarray):
        block = np.array(block)

    if not block.any(): # Handle empty block
        return padded

    try:
        if block.shape[0] > padded.shape[0] or block.shape[1] > padded.shape[1]:
            print(f"Warning: Block shape {block.shape} is larger than padded shape {padded.shape}. Truncating.")
            block = block[:padded.shape[0], :padded.shape[1]]

        padded[:block.shape[0], :block.shape[1]] = block
    except IndexError as e:
        print(f"Error padding block: {e}")
        return np.zeros((5, 5), dtype=np.float32)
    return padded


class BlockBlastEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 4}

    def __init__(self, render_mode=None):
        super().__init__()
        self.board_size = GRID_SIZE
        self.game = BlockBlastLogic(GRID_SIZE, SHAPES)

        self.observation_space = spaces.Dict({
            "grid": spaces.Box(0, 1, shape=(self.board_size, self.board_size), dtype=np.int32),
            "blocks_to_place": spaces.Box(0, 1, shape=(3, 5, 5), dtype=np.float32)
        })

        self.action_space = spaces.MultiDiscrete([3, self.board_size, self.board_size]) # (block_idx, row, col)

        self.render_mode = render_mode

    def _get_observation(self):
        grid_obs = self.game.grid.astype(np.float32)

        blocks_obs = np.zeros((3, 5, 5), dtype=np.float32)
        for i, block_obj in enumerate(self.game.blocks_to_place):
            blocks_obs[i] = pad_block(block_obj.shape_array)
        return {"grid": grid_obs, "blocks_to_place": blocks_obs}

    def _get_info(self):
        return {"score": self.game.score, "moves": self.game.moves_count}

    def step(self, action):
        block_idx, row, col = action[0], action[1], action[2]

        success, placed_block, points_earned_this_turn = self.game.place_block(block_idx, row, col)

        reward = 0.0
        terminated = False
        truncated = False # Not used in this environment, but kept for Gymnasium API compliance

        if success:
            reward = float(points_earned_this_turn) # Reward is the total score increase from game_logic
            
            # After placing a block, if blocks_to_place is empty, generate new blocks
            if not self.game.blocks_to_place:
                self.game.generate_new_blocks()

            # Check for game over (no more valid moves)
            if self.game.check_game_over():
                terminated = True
                reward -= 500.0 # Large penalty for game over.
        else: # Invalid placement
            reward = -10.0 # Penalty for invalid move (e.g., trying to place out of bounds or overlapping)

        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """Resets the environment to a new game."""
        super().reset(seed=seed)
        self.game = BlockBlastLogic(GRID_SIZE, SHAPES)
        observation = self._get_observation()
        info = self._get_info() # Ensure initial info is consistent with game state
        return observation, info

    def render(self):
        """Renders the environment using Pygame."""
        pass 

    def close(self):
        """Closes the environment."""
        if self.render_mode == "human":
            pygame.quit()
            sys.exit()

if __name__ == "__main__":
    print("Running a simple test of the BlockBlastEnv...")
    env = BlockBlastEnv()
    obs, info = env.reset()
    print("Observation space:", env.observation_space)
    print("Initial observation shapes:", {k: v.shape for k, v in obs.items()})
    
    # Take a random step
    action = env.action_space.sample()
    print("Taking a random action:", action)
    observation, reward, terminated, truncated, info = env.step(action)
    print(f"Step result: Reward={reward}, Terminated={terminated}, Truncated={truncated}, Info={info}")
    
    env.close()
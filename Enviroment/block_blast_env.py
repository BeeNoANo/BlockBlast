# block_blast_env.py
import pygame
import sys
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from Helper.game_logic import BlockBlastLogic, GRID_SIZE, SHAPES

def pad_block(block):
    """
    Pads a block matrix to a fixed size of 5x5.
    This size is chosen to accommodate the largest block shape (e.g., 1x5 line).
    """
    # CHỈNH KÍCH THƯỚC TỪ (4, 4) THÀNH (5, 5) để phù hợp với block lớn nhất
    padded = np.zeros((5, 5), dtype=np.float32)
    
    if not isinstance(block, np.ndarray):
        block = np.array(block)

    if not block.any(): # Handle empty block
        return padded

    try:
        # Check to ensure the block fits within the padded array
        if block.shape[0] > padded.shape[0] or block.shape[1] > padded.shape[1]:
            # This should not happen, but it's a good safeguard.
            print(f"Warning: Block shape {block.shape} is larger than padded shape {padded.shape}. Truncating.")
            # Truncate the block if it's too big to prevent errors.
            block = block[:padded.shape[0], :padded.shape[1]]

        padded[:block.shape[0], :block.shape[1]] = block
    except IndexError as e:
        print(f"IndexError in pad_block: {e}")
        print(f"Block shape: {block.shape}")
    except Exception as e:
        print(f"An unexpected error occurred in pad_block: {e}")

    return padded

class BlockBlastEnv(gym.Env):
    """
    A custom Gymnasium environment for the Block Blast game.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        super().__init__()
        self.game = BlockBlastLogic(GRID_SIZE, SHAPES)
        
        # Define action space:
        # A single action represents placing a specific block at a specific position.
        # Action space size = 3 blocks * (GRID_SIZE * GRID_SIZE positions)
        self.action_space = spaces.Discrete(3 * GRID_SIZE * GRID_SIZE)
        
        # Define observation space (SỬA KÍCH THƯỚC KHỐI TỪ 4x4 -> 5x5):
        # The observation is the game grid and 3 padded blocks.
        self.observation_space = spaces.Dict({
            "board": spaces.Box(low=0, high=1, shape=(GRID_SIZE, GRID_SIZE), dtype=np.float32),
            "block_0": spaces.Box(low=0, high=1, shape=(5, 5), dtype=np.float32),
            "block_1": spaces.Box(low=0, high=1, shape=(5, 5), dtype=np.float32),
            "block_2": spaces.Box(low=0, high=1, shape=(5, 5), dtype=np.float32),
        })

    def _get_observation(self):
        """Returns the current observation."""
        # --- CODE ĐÃ SỬA VÀ HOÀN THIỆN ---
        # Get padded grid for the 3 blocks to place
        blocks_padded = [pad_block(block.grid) for block in self.game.blocks_to_place]
        
        # Fill with zeros if there are less than 3 blocks
        while len(blocks_padded) < 3:
            blocks_padded.append(pad_block([])) # Pad with empty block
            
        obs = {
            "board": self.game.grid.astype(np.float32),
            "block_0": blocks_padded[0],
            "block_1": blocks_padded[1],
            "block_2": blocks_padded[2],
        }
        return obs
        # --- KẾT THÚC CODE ĐÃ SỬA ---

    def step(self, action):
        """Performs a single step in the environment based on an action."""
        initial_score = self.game.score
        
        # Decode the action
        block_idx = action // (GRID_SIZE * GRID_SIZE)
        pos_flat = action % (GRID_SIZE * GRID_SIZE)
        r = pos_flat // GRID_SIZE
        c = pos_flat % GRID_SIZE
        
        reward = 0.0
        
        # Check if the placement is valid
        is_valid = self.game.is_valid_placement(block_idx, r, c)
        
        if is_valid:
            # Place the block and get the score from this move
            _, placed_score = self.game.place_block(block_idx, r, c)
            reward += placed_score # Reward for placing and clearing lines
            
            # Check for new blocks to place
            if not self.game.blocks_to_place:
                self.game.generate_new_blocks()

            # Check if the game is over after the move
            terminated = self.game.check_game_over()
        else:
            # Invalid move, apply a penalty
            reward -= 50.0  # Big penalty for an invalid move
            terminated = False
            
        truncated = False # We don't truncate based on steps here
        
        # Get the new observation and info
        observation = self._get_observation()
        info = {"score": self.game.score, "moves": self.game.moves_count}
        
        # If game is over, add a large penalty to strongly discourage losing
        if terminated:
            reward -= 500.0
        
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """Resets the environment to a new game."""
        super().reset(seed=seed)
        self.game = BlockBlastLogic(GRID_SIZE, SHAPES)
        observation = self._get_observation()
        info = {"score": 0, "moves": 0}
        
        return observation, info

    def render(self):
        """Renders the environment using Pygame."""
        pass # This can be left empty as we use run_rl_agent.py for visualization.

    def close(self):
        """Closes the environment."""
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
    obs, reward, terminated, truncated, info = env.step(action)
    print("Reward for this action:", reward)
    print("New observation shapes:", {k: v.shape for k, v in obs.items()})
    
    env.close()
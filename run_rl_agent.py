# run_rl_agent.py

import pygame
import sys
from stable_baselines3 import PPO
from Enviroment.block_blast_env import BlockBlastEnv
from Helper.game_logic import GRID_SIZE, CELL_SIZE, SHAPES
import numpy as np

# Pygame Initialization
pygame.init()
pygame.font.init()

# Display Configuration
WIDTH = GRID_SIZE * CELL_SIZE
HEIGHT = GRID_SIZE * CELL_SIZE + 200
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Block Blast RL Agent")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (150, 150, 150)
LIGHT_GRAY = (200, 200, 200)
# Simple color for blocks
BLOCK_COLOR = (255, 100, 100)
UPCOMING_BLOCK_COLOR = (150, 150, 255)

def draw_grid(screen, grid):
    """Draws the main game grid."""
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            color_index = grid[r][c]
            color = LIGHT_GRAY if color_index == 0 else BLOCK_COLOR
            pygame.draw.rect(screen, color, (c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE))
            pygame.draw.rect(screen, GRAY, (c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE), 1)

def draw_blocks_to_place(screen, blocks_to_place):
    """Draws the next blocks at the bottom of the screen."""
    y_offset = GRID_SIZE * CELL_SIZE + 20
    x_offset = 20
    for block in blocks_to_place:
        for r_idx, row_data in enumerate(block):
            for c_idx, cell_value in enumerate(row_data):
                if cell_value == 1:
                    pygame.draw.rect(screen, UPCOMING_BLOCK_COLOR, (x_offset + c_idx * CELL_SIZE, y_offset + r_idx * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                    pygame.draw.rect(screen, GRAY, (x_offset + c_idx * CELL_SIZE, y_offset + r_idx * CELL_SIZE, CELL_SIZE, CELL_SIZE), 1)
        
        # Move to the next block's position
        if block:
            x_offset += max(len(row) for row in block) * CELL_SIZE + 20
        else: # Handle empty block list
            x_offset += CELL_SIZE + 20

def draw_score(screen, score):
    """Draws the current score on the screen."""
    font = pygame.font.Font(None, 36)
    text = font.render(f"Score: {score}", True, WHITE)
    screen.blit(text, (10, HEIGHT - 50))

def main(model_path):
    """Main loop to run the trained RL agent."""
    
    # Load the trained model
    try:
        # PPO.load automatically adds the .zip extension
        model = PPO.load(model_path)
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}.zip' not found.")
        print("Please check the 'models' directory and ensure the file exists.")
        print("Example usage: python run_rl_agent.py models/block_blast_ppo_model_100000_steps")
        sys.exit()

    # Create the environment for inference (the AI's playground)
    env = BlockBlastEnv()
    
    # Reset the environment to start a new game
    obs, info = env.reset()
    
    clock = pygame.time.Clock()
    
    done = False
    
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                
        # The agent predicts the next action using the loaded model
        # `deterministic=True` means the agent will always choose the best action it knows.
        action, _states = model.predict(obs, deterministic=True)
        
        # Take the action in the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        done = terminated or truncated
        
        # Manually render the game state for visualization
        SCREEN.fill(BLACK)
        draw_grid(SCREEN, env.game.grid)
        draw_blocks_to_place(SCREEN, env.game.blocks_to_place)
        draw_score(SCREEN, env.game.score)
        pygame.display.flip()

        # Control the speed of the simulation for visualization
        # The AI will make a move every 100ms (10 FPS)
        clock.tick(10) 

    print("Game finished. Final Score:", info['score'])
    env.close()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    # Get the model path from the command line argument, if provided.
    # Otherwise, use the default name of the final model.
    import sys
    model_to_load = "final_ppo_block_blast_model"  # Default final model name
    
    if len(sys.argv) > 1:
        model_to_load = sys.argv[1]
    
    # Run the main function with the specified model path
    main(model_to_load)
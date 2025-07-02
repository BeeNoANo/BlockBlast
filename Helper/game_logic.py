# game_logic.py

import random
import numpy as np

# Game Configuration
GRID_SIZE = 8
CELL_SIZE = 50
SHAPES = {
    # Khối 1x1
    'dot': [[1]],

    # Hình chữ nhật 2x3
    'rect_2x3_v': [[1, 1], [1, 1], [1, 1]], # Dọc
    'rect_2x3_h': [[1, 1, 1], [1, 1, 1]], # Ngang

    # Hình vuông
    'square_2x2': [[1, 1], [1, 1]],
    'square_3x3': [[1, 1, 1], [1, 1, 1], [1, 1, 1]],

    # Chữ L 3x3 (4 hướng xoay)
    'L_3x3_0': [[1, 0, 0], [1, 0, 0], [1, 1, 1]],
    'L_3x3_90': [[1, 1, 1], [1, 0, 0], [1, 0, 0]],
    'L_3x3_180': [[1, 1, 1], [0, 0, 1], [0, 0, 1]],
    'L_3x3_270': [[0, 0, 1], [0, 0, 1], [1, 1, 1]],

    # Thanh thẳng
    'line_1_h': [[1]],
    'line_2_h': [[1, 1]],
    'line_3_h': [[1, 1, 1]],
    'line_4_h': [[1, 1, 1, 1]],
    'line_5_h': [[1, 1, 1, 1, 1]],

    'line_1_v': [[1]],
    'line_2_v': [[1], [1]],
    'line_3_v': [[1], [1], [1]],
    'line_4_v': [[1], [1], [1], [1]],
    'line_5_v': [[1], [1], [1], [1], [1]],

    # Chữ thập
    'cross_3x3': [[0, 1, 0], [1, 1, 1], [0, 1, 0]],

    # Thanh chữ L
    'small_L_2x2_0': [[1, 0], [1, 1]],
    'small_L_2x2_90': [[1, 1], [1, 0]],
    
    # Chữ T
    'T_shape': [[1, 1, 1], [0, 1, 0]],
}

class Block:
    """Represents a block with its shape (grid) and name."""
    def __init__(self, name, grid):
        self.name = name
        self.grid = np.array(grid) # Use numpy array for easier handling

class BlockBlastLogic:
    """Handles the core logic of the Block Blast game."""
    def __init__(self, grid_size, shapes):
        self.grid_size = grid_size
        self.shapes = shapes
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.blocks_to_place = []
        self.score = 0
        self.moves_count = 0
        self.generate_new_blocks()

    def get_board_state(self):
        """Returns the current state of the board."""
        return self.grid.copy()

    def get_blocks_to_place_from_names(self, names):
        """Returns a list of Block objects from a list of shape names."""
        return [Block(name, self.shapes[name]) for name in names]

    def generate_new_blocks(self):
        """
        Generates a new set of 3 random blocks, ensuring at least one can be placed.
        The blocks in the set can now be duplicates.
        """
        if not self.blocks_to_place:
            all_shape_names = list(self.shapes.keys())
            placeable_shapes = []

            # 1. Find all shapes that can be placed on the current grid.
            for name, grid_data in self.shapes.items():
                grid_array = np.array(grid_data)
                block_h, block_w = grid_array.shape
                
                found_placement = False
                # Iterate through all possible positions
                for r in range(self.grid_size - block_h + 1):
                    for c in range(self.grid_size - block_w + 1):
                        # Check for overlaps
                        is_valid = True
                        for br, row_data in enumerate(grid_array):
                            for bc, cell_val in enumerate(row_data):
                                if cell_val == 1 and self.grid[r + br][c + bc] != 0:
                                    is_valid = False
                                    break
                            if not is_valid:
                                break
                        if is_valid:
                            placeable_shapes.append(name)
                            found_placement = True
                            break # Found one valid spot for this shape, move to the next shape.
                    if found_placement:
                        break # Exit the outer loop too.

            # 2. Generate the new block set.
            new_blocks_names = []
            
            if placeable_shapes:
                # Case A: At least one block can be placed.
                # Pick one guaranteed placeable block.
                guaranteed_block_name = random.choice(placeable_shapes)
                new_blocks_names.append(guaranteed_block_name)
                
                # Pick the remaining 2 blocks randomly from the entire pool.
                # As per your clarification, duplicates are now allowed.
                while len(new_blocks_names) < 3:
                    chosen_name = random.choice(all_shape_names)
                    new_blocks_names.append(chosen_name)
            else:
                # Case B: No blocks can be placed (game over condition).
                # We still generate 3 random blocks, but the game will terminate
                # at the next `check_game_over()` call in the environment.
                new_blocks_names = random.sample(all_shape_names, 3)
                print("Warning: No shape from the pool can be placed. This set of blocks will likely lead to a 'game over' state.")

            # Shuffle the list to randomize the order of the blocks
            random.shuffle(new_blocks_names)
            
            # 3. Create the Block objects and assign them
            self.blocks_to_place = self.get_blocks_to_place_from_names(new_blocks_names)

    def is_valid_placement(self, block_index, row, col):
        """
        Checks if a block can be placed at a given position.
        Args:
            block_index (int): Index of the block in self.blocks_to_place.
            row (int): The starting row for placement.
            col (int): The starting column for placement.
        """
        # Get the block's grid from the list of current blocks to place
        if block_index >= len(self.blocks_to_place):
            return False # Index out of bounds

        block_grid = self.blocks_to_place[block_index].grid
        
        # Get block dimensions
        block_h, block_w = block_grid.shape
        
        # Check if the block fits within the grid boundaries from the starting position
        if row + block_h > self.grid_size or col + block_w > self.grid_size:
            return False

        # Check for overlaps with existing blocks on the grid
        for r_idx, row_data in enumerate(block_grid):
            for c_idx, cell_value in enumerate(row_data):
                if cell_value == 1:
                    # Check if the corresponding grid cell is already occupied
                    if self.grid[row + r_idx][col + c_idx] != 0:
                        return False  # Overlap found
        
        return True  # No issues, placement is valid

    def place_block(self, block_index, row, col):
        """Places a block on the grid and removes it from the list."""
        if not self.is_valid_placement(block_index, row, col):
            return False, 0 

        block_to_place = self.blocks_to_place.pop(block_index)
        block_grid = block_to_place.grid
        
        # Update the grid with the block's shape
        for r_idx, row_data in enumerate(block_grid):
            for c_idx, cell_value in enumerate(row_data):
                if cell_value == 1:
                    self.grid[row + r_idx][col + c_idx] = 1
        
        self.moves_count += 1
        
        # Calculate score for placing the block
        placed_score = np.sum(block_grid)
        self.score += placed_score
        
        cleared_score = self.clear_lines()

        return True, placed_score + cleared_score

    def clear_lines(self):
        """Clears completed rows and columns and updates the score."""
        cleared_count = 0
        
        # Find rows to clear
        rows_to_clear = [r for r in range(self.grid_size) if all(self.grid[r] != 0)]
        
        # Find columns to clear
        cols_to_clear = [c for c in range(self.grid_size) if all(self.grid[r][c] != 0 for r in range(self.grid_size))]
        
        # Clear rows
        for r in rows_to_clear:
            self.grid[r] = np.zeros(self.grid_size, dtype=int)
            cleared_count += 1
            
        # Clear columns
        for c in cols_to_clear:
            for r in range(self.grid_size):
                self.grid[r][c] = 0
            cleared_count += 1
            
        cleared_score = 0
        if cleared_count > 0:
            # Score for cleared cells
            cleared_score = cleared_count * self.grid_size
            if cleared_count > 1: # Combo bonus
                cleared_score += (cleared_count - 1) * self.grid_size * 2
            
            self.score += cleared_score

        return cleared_score

    def check_game_over(self):
        """Checks if there's any valid move left for the current blocks."""
        if not self.blocks_to_place:
            # If no blocks left, it's not game over, new blocks will be generated.
            return False 

        for block_idx, block in enumerate(self.blocks_to_place):
            for r in range(self.grid_size):
                for c in range(self.grid_size):
                    # Use the corrected validation method
                    if self.is_valid_placement(block_idx, r, c):
                        return False  # A valid move exists, so the game is not over
        
        return True  # No valid moves found for any of the current blocks
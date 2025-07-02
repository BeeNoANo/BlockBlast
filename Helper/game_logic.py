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

    # Các khối thẳng
    'line_1x4_v': [[1], [1], [1], [1]],
    'line_1x4_h': [[1, 1, 1, 1]],
    'line_1x5_v': [[1], [1], [1], [1], [1]],
    'line_1x5_h': [[1, 1, 1, 1, 1]],

    # Các khối Z
    'Z_2x3_0': [[1, 1, 0], [0, 1, 1]],
    'Z_2x3_90': [[0, 1], [1, 1], [1, 0]],

    # Các khối T
    'T_3x2_0': [[1, 1, 1], [0, 1, 0]],
    'T_3x2_90': [[0, 1], [1, 1], [0, 1]],

    # Các khối tùy chỉnh khác
    'cross_3x3': [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
    'plus': [[0,1,0],[1,1,1],[0,1,0]]
}

class Block:
    def __init__(self, shape_key, shape_array):
        self.shape_key = shape_key
        self.shape_array = np.array(shape_array, dtype=int)
        self.rows = self.shape_array.shape[0]
        self.cols = self.shape_array.shape[1]

class BlockBlastLogic:
    def __init__(self, grid_size, shapes):
        self.grid_size = grid_size
        self.grid = np.zeros((grid_size, grid_size), dtype=int)
        self.shapes = shapes
        self.score = 0
        self.blocks_to_place = [] # 3 blocks available to the player
        self.moves_count = 0
        self.consecutive_clears_count = 0 # Initialize consecutive clears count

        self.generate_new_blocks() # Generate initial blocks

    def generate_new_blocks(self):
        self.blocks_to_place = []
        all_shape_keys = list(self.shapes.keys())
        while True:
            current_set_of_blocks = []
            for _ in range(3):
                chosen_key = random.choice(all_shape_keys)
                new_block = Block(chosen_key, self.shapes[chosen_key])
                current_set_of_blocks.append(new_block)
            
            # Check if at least one block in this set is placeable
            has_placeable_block_in_set = False
            for block in current_set_of_blocks:
                for r in range(self.grid_size):
                    for c in range(self.grid_size):
                        if self._is_valid_placement(block, r, c, self.grid):
                            has_placeable_block_in_set = True
                            break
                    if has_placeable_block_in_set:
                        break
                if has_placeable_block_in_set:
                    break
            
            if has_placeable_block_in_set:
                self.blocks_to_place = current_set_of_blocks
                break # Found a valid set of 3 blocks, exit loop

    def _is_valid_placement(self, block, row, col, current_grid):
        if row + block.rows > self.grid_size or col + block.cols > self.grid_size:
            return False # Out of bounds

        for r_offset in range(block.rows):
            for c_offset in range(block.cols):
                if block.shape_array[r_offset][c_offset] == 1:
                    if current_grid[row + r_offset][col + c_offset] != 0:
                        return False # Overlapping
        return True

    def get_all_valid_placements(self):
        valid_placements = []
        for block_idx, block in enumerate(self.blocks_to_place):
            for r in range(self.grid_size):
                for c in range(self.grid_size):
                    if self._is_valid_placement(block, r, c, self.grid):
                        valid_placements.append((block_idx, r, c))
        return valid_placements

    def place_block(self, block_idx, row, col):
        if not (0 <= block_idx < len(self.blocks_to_place)):
            return False, None, 0 # Invalid block index, return 0 points

        block_to_place = self.blocks_to_place[block_idx]

        if not self._is_valid_placement(block_to_place, row, col, self.grid):
            return False, None, 0 # Invalid placement, return 0 points

        # Create a deep copy of the grid to apply changes
        new_grid = np.copy(self.grid)
        for r_offset in range(block_to_place.rows):
            for c_offset in range(block_to_place.cols):
                if block_to_place.shape_array[r_offset][c_offset] == 1:
                    new_grid[row + r_offset][col + c_offset] = 1 # Place the block

        self.grid = new_grid # Update the game grid
        self.moves_count += 1

        # Remove the placed block from blocks_to_place
        placed_block_obj = self.blocks_to_place.pop(block_idx)

        # Calculate points for placing the block
        points_from_placement = placed_block_obj.rows * placed_block_obj.cols

        # Clear lines and get points from clearing
        points_from_clearing = self._clear_lines()

        # Update consecutive clears count and apply multiplier
        if points_from_clearing > 0:
            self.consecutive_clears_count += 1
        else:
            self.consecutive_clears_count = 0 # Reset if no lines cleared

        total_points_this_turn = points_from_placement
        if points_from_clearing > 0:
            multiplier = 1.0
            if self.consecutive_clears_count > 1:
                # Example multiplier: 1.0 for 1st consecutive, 1.5 for 2nd, 2.0 for 3rd, etc.
                multiplier = 1.0 + (self.consecutive_clears_count - 1) * 0.5
            total_points_this_turn += points_from_clearing * multiplier
        
        self.score += total_points_this_turn

        return True, placed_block_obj, total_points_this_turn # Return success, block, and total points earned

    def _clear_lines(self):
        rows_to_clear = []
        cols_to_clear = []

        # Check rows
        for r in range(self.grid_size):
            if np.all(self.grid[r] == 1):
                rows_to_clear.append(r)

        # Check columns
        for c in range(self.grid_size):
            if np.all(self.grid[:, c] == 1):
                cols_to_clear.append(c)

        cleared_count = len(rows_to_clear) + len(cols_to_clear)
        cleared_score = 0

        if cleared_count > 0:
            # Score for cleared cells based on user's prompt: lines_cleared * 10 + combo_bonus
            base_line_clear_points = cleared_count * 10
            combo_bonus_points = 0
            if cleared_count > 1: # Combo bonus for clearing more than one line/column
                combo_bonus_points = (cleared_count - 1) * 5
            cleared_score = base_line_clear_points + combo_bonus_points

            # Clear rows
            for r in rows_to_clear:
                self.grid[r] = np.zeros(self.grid_size, dtype=int)

            # Clear columns
            for c in cols_to_clear:
                for r_idx in range(self.grid_size):
                    self.grid[r_idx][c] = 0
            
            return cleared_score
        return 0

    def check_game_over(self):
        """Checks if there's any valid move left for the current blocks."""
        if not self.blocks_to_place:
            # If no blocks left, it's not game over, new blocks will be generated.
            return False

        # If there are no valid placements for any of the current blocks, game is over.
        if not self.get_all_valid_placements():
            return True
        return False

# This is a helper function to check if the file is imported correctly.
if __name__ == "__main__":
    print("Running a simple test of the BlockBlastLogic...")
    game = BlockBlastLogic(GRID_SIZE, SHAPES)
    print(f"Initial score: {game.score}")
    print(f"Blocks to place: {[b.shape_key for b in game.blocks_to_place]}")
    print(f"Initial grid:\n{game.grid}")

    # Example: Try to place the first block at (0,0)
    if game.blocks_to_place:
        test_block_idx = 0
        test_row, test_col = 0, 0
        success, placed_block, points_earned = game.place_block(test_block_idx, test_row, test_col)

        if success:
            print(f"\nSuccessfully placed {placed_block.shape_key} at ({test_row}, {test_col})")
            print(f"Points earned this turn: {points_earned}")
            print(f"New score: {game.score}")
            print(f"New grid:\n{game.grid}")
            print(f"Blocks remaining: {[b.shape_key for b in game.blocks_to_place]}")
        else:
            print("Invalid placement. Try again.")
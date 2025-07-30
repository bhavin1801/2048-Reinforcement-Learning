import random
import numpy as np

UP, RIGHT, DOWN, LEFT = range(4)
action_name = ["UP", "RIGHT", "DOWN", "LEFT"]

class IllegalAction(Exception):
    pass

class GameOver(Exception):
    pass

class Board:
    def __init__(self, board=None):
        self.board = np.zeros((16), dtype=int) if board is None else np.array(board, dtype=int)
        self.spawn_tile()   

    def spawn_tile(self):
        empty = [i for i in range(16) if self.board[i] == 0]
        if not empty:
            return
        i = random.choice(empty)
        self.board[i] = 2 if random.random() < 0.9 else 4

    def copyboard(self):
        return np.copy(self.board).reshape((16))

    def rotate_board(self, times):
        return np.rot90(self.board, -times)

    def act(self, action):
        rotated = np.rot90(self.board.reshape(4, 4), -action)
        moved, score = self._move_left(rotated)
        if moved is None:
            raise IllegalAction()
        
        self.board = np.rot90(moved, action).flatten()
        return score

    def _move_left(self, board):
        score = 0
        new_board = np.zeros_like(board)
        moved = False

        for i in range(4):
            line = board[i]
            new_line = [x for x in line if x != 0]
            merged_line = []
            skip = False

            j = 0
            while j < len(new_line):
                if not skip and j + 1 < len(new_line) and new_line[j] == new_line[j + 1]:
                    merged_line.append(new_line[j] + 1)
                    score += 2 ** (new_line[j] + 1)
                    skip = True
                    moved = True
                else:
                    if skip:
                        skip = False
                    else:
                        merged_line.append(new_line[j])
                j += 1

            merged_line += [0] * (4 - len(merged_line))
            new_board[i] = merged_line

            if not np.array_equal(line, merged_line):
                moved = True

        if moved:
            return new_board, score
        else:
            return None, None

                    
    def is_game_over(self):
        if 0 in self.board:
            return False
        for a in [UP, RIGHT, DOWN, LEFT]:
            try:
                temp = Board(self.board.copy())
                temp.act(a)
                return False
            except IllegalAction:
                continue
        return True
    
    def display(self):
        board_2d = self.board.reshape((4, 4))
        print("+----" * 4 + "+")
        for row in board_2d:
            print("|" + "|".join(f"{2**val if val > 0 else 0:^4}" for val in row) + "|")
            print("+----" * 4 + "+")
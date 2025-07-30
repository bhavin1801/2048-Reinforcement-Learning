import numpy as np
import random
import pickle
from collections import defaultdict

# Define Game and Network (same as your previous code)
class Game2048:
    def __init__(self):
        self.board = np.zeros((4, 4), dtype=int)
        self.score = 0
        self.reset()

    def reset(self):
        self.board[:] = 0
        self.score = 0
        self.add_tile()
        self.add_tile()

    def add_tile(self):
        empty = list(zip(*np.where(self.board == 0)))
        if empty:
            i, j = random.choice(empty)
            self.board[i][j] = 2 if random.random() < 0.9 else 4

    def compress(self, row):
        row = row[row != 0]
        return np.pad(row, (0, 4 - len(row)), 'constant')

    def merge(self, row):
        for i in range(3):
            if row[i] and row[i] == row[i + 1]:
                row[i] *= 2
                self.score += row[i]
                row[i + 1] = 0
        return row

    def move_left(self):
        moved = False
        for i in range(4):
            original = self.board[i].copy()
            row = self.compress(self.board[i])
            row = self.merge(row)
            row = self.compress(row)
            self.board[i] = row
            if not np.array_equal(original, row):
                moved = True
        if moved:
            self.add_tile()
        return moved

    def move_right(self):
        self.board = np.fliplr(self.board)
        moved = self.move_left()
        self.board = np.fliplr(self.board)
        return moved

    def move_up(self):
        self.board = np.rot90(self.board, -1)
        moved = self.move_left()
        self.board = np.rot90(self.board)
        return moved

    def move_down(self):
        self.board = np.rot90(self.board)
        moved = self.move_left()
        self.board = np.rot90(self.board, -1)
        return moved

    def is_game_over(self):
        for fn in [self.move_left, self.move_right, self.move_up, self.move_down]:
            copy = self.board.copy()
            if fn():
                self.board = copy
                return False
            self.board = copy
        return True

    def do_move(self, a):
        return [self.move_left, self.move_right, self.move_up, self.move_down][a]()

# N-Tuple Network
class NTupleNetwork:
    def __init__(self, tuples):
        self.tuples = tuples
        self.lut = [defaultdict(float) for _ in tuples]

    def get_features(self, board):
        features = []
        for idx, tup in enumerate(self.tuples):
            key = tuple(int(np.log2(board[i][j])) if board[i][j] else 0 for (i, j) in tup)
            features.append((idx, key))
        return features

    def evaluate(self, board):
        features = self.get_features(board)
        return sum(self.lut[idx][key] for idx, key in features)

# Load model
with open("ntuple_model.pkl", "rb") as f:
    saved_lut = pickle.load(f)

# Same tuples as training
tuples = [
    [(0,0), (0,1), (0,2), (0,3)],
    [(1,0), (1,1), (1,2), (1,3)],
    [(2,0), (2,1), (2,2), (2,3)],
    [(3,0), (3,1), (3,2), (3,3)],
    [(0,0), (1,0), (2,0), (3,0)],
    [(0,1), (1,1), (2,1), (3,1)],
    [(0,2), (1,2), (2,2), (3,2)],
    [(0,3), (1,3), (2,3), (3,3)],
]

ntuple = NTupleNetwork(tuples)
ntuple.lut = saved_lut

game = Game2048()

def get_valid_moves(game):
    valid = []
    for a in range(4):
        temp = game.board.copy()
        moved = game.do_move(a)
        if moved:
            valid.append((a, temp.copy()))
        game.board = temp
    return valid

steps = 0
while not game.is_game_over():
    valid = get_valid_moves(game)
    if not valid:
        break
    # Greedy move
    best_action = max(valid, key=lambda x: ntuple.evaluate(x[1]))[0]
    game.do_move(best_action)
    steps += 1
    print(f"Step {steps} | Score: {game.score}")
    print(game.board, "\n")

print("Game over!")
print("Final Score:", game.score)
print("Max Tile:", np.max(game.board))
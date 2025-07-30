import numpy as np
import random
from collections import defaultdict
import csv
import pickle

# Game logic
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

    def can_move(self):
        for move_fn in [self.move_left, self.move_right, self.move_up, self.move_down]:
            backup = self.board.copy()
            if move_fn():
                self.board = backup
                return True
            self.board = backup
        return False

    def is_game_over(self):
        return not self.can_move()

    def get_state(self):
        return self.board.copy()

    def do_move(self, action):
        if action == 0:
            return self.move_left()
        elif action == 1:
            return self.move_right()
        elif action == 2:
            return self.move_up()
        elif action == 3:
            return self.move_down()

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

    def update(self, board, td_error, alpha=0.01):
        features = self.get_features(board)
        for idx, key in features:
            self.lut[idx][key] += alpha * td_error

# Agent Training
def get_valid_moves(game):
    valid = []
    for a in range(4):
        temp = game.get_state()
        moved = game.do_move(a)
        if moved:
            valid.append(a)
        game.board = temp
    return valid

def train_agent(episodes=1000, alpha=0.01, gamma=0.99):
    # Define simple tuples
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
    game = Game2048()

    # Open CSV for logging
    log_file = open("training_log.csv", "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow(["Episode", "Score", "MaxTile", "Steps"])

    for ep in range(episodes):
        game.reset()
        steps = 0
        while not game.is_game_over():
            state = game.get_state()
            value = ntuple.evaluate(state)

            valid_moves = get_valid_moves(game)
            action = random.choice(valid_moves)
            game.do_move(action)

            reward = np.log2(game.score + 1)
            next_state = game.get_state()
            next_value = ntuple.evaluate(next_state)

            td_error = reward + gamma * next_value - value
            ntuple.update(state, td_error, alpha)

            steps += 1

        writer.writerow([ep+1, game.score, np.max(game.board), steps])
        print(f"Episode {ep+1:4d} | Score: {game.score:5d} | Max Tile: {np.max(game.board):4d} | Steps: {steps}")

    log_file.close()
    return ntuple

if __name__ == "__main__":
    trained_model = train_agent(episodes=1000)

    # Save model to a file
    with open("ntuple_model.pkl", "wb") as f:
        pickle.dump(trained_model.lut, f)

    print("✅ Trained model saved as ntuple_model.pkl")

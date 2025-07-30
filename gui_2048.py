import tkinter as tk
import numpy as np
import time
from game import Board, action_name, UP, RIGHT, DOWN, LEFT
from agent import nTupleNewrok
from game import IllegalAction, GameOver

CELL_COLORS = {
    0: "#cdc1b4",
    2: "#eee4da",
    4: "#ede0c8",
    8: "#f2b179",
    16: "#f59563",
    32: "#f67c5f",
    64: "#f65e3b",
    128: "#edcf72",
    256: "#edcc61",
    512: "#edc850",
    1024: "#edc53f",
    2048: "#edc22e",
}

class Game2048GUI:
    def __init__(self, agent):
        self.agent = agent
        self.window = tk.Tk()
        self.window.title("2048 Agent")
        self.board = Board()
        self.size = 4
        self.cell_size = 100
        self.margin = 10
        self.canvas = tk.Canvas(
            self.window,
            width=self.size*self.cell_size + 2*self.margin,
            height=self.size*self.cell_size + 2*self.margin,
            bg="#bbada0"
        )
        self.canvas.pack()
        self.draw_board()
        self.window.after(1000, self.agent_play)
        self.window.mainloop()

    def draw_board(self):
        self.canvas.delete("all")
        board_2d = self.board.board.reshape((self.size, self.size))
        for i in range(self.size):
            for j in range(self.size):
                value = 2 ** board_2d[i][j] if board_2d[i][j] > 0 else 0
                color = CELL_COLORS.get(value, "#3c3a32")
                x0 = self.margin + j*self.cell_size
                y0 = self.margin + i*self.cell_size
                x1 = x0 + self.cell_size
                y1 = y0 + self.cell_size
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="#bbada0")
                if value > 0:
                    self.canvas.create_text(
                        (x0+x1)//2, (y0+y1)//2,
                        text=str(value),
                        font=("Helvetica", 32, "bold"),
                        fill="#776e65"
                    )

    def agent_play(self):
        try:
            # Check if game is already over before trying to act
            if self.board.is_game_over():
                raise GameOver("No more moves possible") # Raise your GameOver for consistent handling

            a_best = self.agent.best_action(self.board.board)
            self.board.act(a_best) # This might raise IllegalAction
            self.board.spawn_tile()
            self.draw_board()
            self.window.after(300, self.agent_play)
        except (IllegalAction, GameOver): # Catch both, now that we might raise GameOver
            # If an IllegalAction happened or we explicitly raised GameOver
            if self.board.is_game_over(): # Confirm it's truly over
                self.draw_board()
                self.canvas.create_text(
                    self.size*self.cell_size//2 + self.margin,
                    self.size*self.cell_size//2 + self.margin,
                    text="Game Over!",
                    font=("Helvetica", 32, "bold"),
                    fill="red"
                )
            else:
                # This should ideally not happen if best_action always returns a legal move
                # But if it does, the agent picked an illegal move, keep trying
                # We simply don't schedule a new agent_play, which ends the GUI's simulation.
                # A better agent would ensure it picks a legal move.
                self.window.after(300, self.agent_play) # Re-schedule to try again if not truly game over

if __name__ == "__main__":
    # Import your TUPLES definition from main.py or define it here
    TUPLES = [
        [(0, 0), (0, 1), (0, 2), (0, 3)],
        [(1, 0), (1, 1), (1, 2), (1, 3)],
        [(2, 0), (2, 1), (2, 2), (2, 3)],
        [(3, 0), (3, 1), (3, 2), (3, 3)],
        [(0, 0), (1, 0), (2, 0), (3, 0)],
        [(0, 1), (1, 1), (2, 1), (3, 1)],
        [(0, 2), (1, 2), (2, 2), (3, 2)],
        [(0, 3), (1, 3), (2, 3), (3, 3)],
    ]
    agent = nTupleNewrok(TUPLES)
    Game2048GUI(agent)
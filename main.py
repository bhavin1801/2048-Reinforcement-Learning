import numpy as np
import pickle
import os
from pathlib import Path
from game import Board, IllegalAction, GameOver
from agent import nTupleNewrok
from collections import namedtuple

Transition = namedtuple("Transition", "s, a, r, s_after, s_next")
Gameplay = namedtuple("Gameplay", "transition_history game_reward max_tile")

def play(agent, board, spawn_random_tile=False, display=False, delay=0.3, epsilon=0.1):
    import time
    b = Board(board)
    r_game = 0
    transition_history = []

    if spawn_random_tile:
        b.spawn_tile()
        b.spawn_tile()

    while True:
        if display:
            b.display()
            time.sleep(delay)

        a_best = agent.best_action(b.board, epsilon=epsilon)

        s = b.copyboard()
        try:
            r = b.act(a_best)
            r_game += r
            s_after = b.copyboard()
            b.spawn_tile()
            s_next = b.copyboard()
            transition_history.append(
                Transition(s=s, a=a_best, r=r, s_after=s_after, s_next=s_next)
            )
        except IllegalAction:
            if b.is_game_over():
                break
            continue
        except GameOver:
            break

    gp = Gameplay(
        transition_history=transition_history,
        game_reward=r_game,
        max_tile=2 ** max(b.board.flatten()),
    )
    learn_from_gameplay(agent, gp)
    return gp

def learn_from_gameplay(agent, gameplay, alpha=0.01):
    for transition in reversed(gameplay.transition_history):
        agent.learn(
            s=transition.s,
            a=transition.a,
            r=transition.r,
            s_after=transition.s_after,
            s_next=transition.s_next,
            alpha=alpha
        )

def train(agent, episodes=10000, display=False, save_path="checkpoints"):
    # ✅ Ensure directory exists using absolute path
    save_path = os.path.abspath(save_path)
    os.makedirs(save_path, exist_ok=True)
    print(f"Model checkpoints will be saved in: {save_path}")

    epsilon = 1.0
    min_epsilon = 0.05
    decay_rate = 0.995
    history = []

    for episode in range(1, episodes + 1):
        print(f"Training episode {episode}/{episodes}...", end="\r")
        board = np.zeros((16), dtype=int)
        gp = play(agent, board, spawn_random_tile=True, display=display, epsilon=epsilon)
        epsilon = max(min_epsilon, epsilon * decay_rate)
        history.append((gp.game_reward, gp.max_tile))

        if episode % 10000 == 0:
            avg_score = np.mean([h[0] for h in history[-50:]])
            max_tile = max([h[1] for h in history[-50:]])
            print(f"\n[Episode {episode}] Avg Score (last 50): {avg_score:.2f}, Max Tile: {max_tile}, Epsilon: {epsilon:.3f}")

        if episode % 10000 == 0:
            model_file = os.path.join(save_path, f"agent_ep{episode}.pkl")
            with open(model_file, "wb") as f:
                pickle.dump(agent, f)

    return history

if __name__ == "__main__":
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
    history = train(agent, episodes=1000, display=False, save_path="checkpoints")

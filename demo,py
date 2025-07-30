from game import Board
from main import load_agent
from pathlib import Path
ngame, agent = load_agent(Path('/Users/hengyue/Documents/1. py/RL 2048/tmp/nTupleNewrok_100157games.pkl'))#Update accordingly

import time
from IPython.display import display, clear_output

b = Board()
nstep = 0
rgame = 0
while True:
    a = agent.best_action(b.board)
    rgame += b.act(a)
    future_r = agent.V(b.board)
    nstep += 1
    b.spawn_tile()
    clear_output(wait=True)
    print('Agent expects to reach score: {:.0f}'.format(rgame + future_r))
    print('Score: {}, Steps: {}, Max tile: {}.'.format(rgame, nstep, 2**max(b.board)))
    print()
    b.display()
    time.sleep(0.1)

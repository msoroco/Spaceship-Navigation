import numpy as np
import random
from simulator import Body, Spaceship, Simulator
from animation import SimAnimation
from replay import Transition, ReplayMemory
from model import DQN
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F

BOX_WIDTH = 20
GRID_RADIUS = 10
DRAW_NEIGHBOURHOOD = False

T = 1000


## demonstration of the simulator interface
if __name__ == '__main__':
    sim = Simulator("./simulations/sim2.json")
    
    #### testing:
    sim.start()
    states = [sim.get_current_frame()]
    
    for i in range(T):
        action = (i % 2) *2 #random.randint(0, 4)
        next_state, reward, terminated = sim.step(action)
        states.append(sim.get_current_frame())
        # print(sim.info())
        if terminated == True:
            # break
            pass

    anim = SimAnimation(sim.bodies, sim.objective, sim.limits, states, len(states), 1, 1, "speed5.5", DRAW_NEIGHBOURHOOD, sim.grid_radius, sim.box_width)

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

T = 10000


## demonstration of the simulator interface
if __name__ == '__main__':
    sim = Simulator("./simulations/sim_big_stride.json")
    
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


# import numpy as np
# import random
# from simulator import Body, Spaceship, Simulator
# from animation import SimAnimation
# from replay import Transition, ReplayMemory
# from model import PolicyNetwork
# import torch

# # Hyperparameters (adjust these based on your environment)
# BOX_WIDTH = 20
# GRID_RADIUS = 10
# DRAW_NEIGHBOURHOOD = False
# T = 1000  # Number of steps per episode

# if __name__ == '__main__':
#     # Initialize the simulator
#     # Test Setup (Dynamic)
#     sim = Simulator(filepath="simulations/sim_big_stride.json")
#     state = sim.start()  # Start the environment
#     state_dim = np.prod(state.shape)  # Flattened size of the state
#     action_dim = sim.info()[1]  # Get number of actions

#     # Initialize the policy network with the dynamic state_dim
#     policy = PolicyNetwork(input_dim=state_dim, output_dim=action_dim)
#     policy.load_state_dict(torch.load("models/policy_network.pth"))
#     policy.eval()  # Set the model to evaluation mode

#     # Start the simulator
#     states = [sim.get_current_frame()]

#     # Run the simulation and use the policy to make decisions
#     for i in range(T):
#         state = torch.tensor(sim.get_current_frame(), dtype=torch.float32).flatten().unsqueeze(0)
#         with torch.no_grad():
#             action_probs = policy(state)
        
#         # Choose action based on the model's output
#         action_dist = torch.distributions.Categorical(action_probs)
#         action = action_dist.sample().item()

#         # Take the action in the simulator
#         next_state, reward, terminated = sim.step(action)
#         states.append(sim.get_current_frame())
        
#         if terminated:
#             break

#     # Animation of the results
#     anim = SimAnimation(sim.bodies, sim.objective, sim.limits, states, len(states), 1, 1, "speed5.5", DRAW_NEIGHBOURHOOD, sim.grid_radius, sim.box_width)

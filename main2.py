import os
import numpy as np
import random
import argparse
import gc
import matplotlib.pyplot as plt
from simulator import Body, Spaceship, Simulator
from animation import SimAnimation
from replay import Transition, ReplayMemory
from model import DQN, FullyConnectedDQN
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F


def draw_heatmap(sim : Simulator, title):
    box_width, limits = sim.get_environment_info()
    _, n_actions = sim.info()
    states = [] 
    for x in range(-limits, limits, box_width):
        for y in range(-limits, limits, box_width):
            states.append(sim.get_current_state([x, y]))
    
    states = torch.tensor(np.array(states), dtype=torch.float, requires_grad=False)
    states = states.to(device)
    Qscores = policy_net(states)
    Qscores = Qscores.detach().cpu().numpy()
    Qscores = Qscores.T.reshape((n_actions, int(np.floor(2*limits/box_width)) * int(np.floor(2*limits/box_width))))
    Qscores = Qscores.reshape((n_actions, int(np.floor(2*limits/box_width)), int(np.floor(2*limits/box_width))))

    # Assuming Qscores is already defined
    fig, axs = plt.subplots(2, 3, sharex=True)

    axs[0, 0].imshow(Qscores[0, :, :])
    axs[0, 0].set_title('Q(s, Up)')
    axs[0, 1].imshow(Qscores[1, :, :])
    axs[0, 1].set_title('Q(s, Down)')
    pcm = axs[0, 2].imshow(np.amax(Qscores, 0))
    axs[0, 2].set_title('Q(s, best_action)')
    axs[1, 0].imshow(Qscores[2, :, :])
    axs[1, 0].set_title('Q(s, Left)')
    axs[1, 1].imshow(Qscores[3, :, :])
    axs[1, 1].set_title('Q(s, Right)')
    axs[1, 2].imshow(Qscores[4, :, :])
    axs[1, 2].set_title('Q(s, nothing)')

    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    fig.colorbar(pcm, cax=cbar_ax)

    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust layout to make space for colorbar
    plt.show()

    plt.show()
    plt.savefig('heatmap_' + title +'.png')


def select_action(state):
    """ the exploration probability starts with a value of `EPS_START` and
        then linearly anneals to `EPS_END` over the course of `EXP_ANNEAL_SAMPLES`
        timesteps."""
    sample = np.random.uniform(0, 1)

    l = np.clip(update_modeling_step/EXP_ANNEAL_SAMPLES, 0, 1)
    eps_threshold = (1 - l) * EPS_START + l * EPS_END

    if TEST or OFFLINE or sample <= eps_threshold:
        with torch.no_grad():
            return policy_net(torch.tensor(state, dtype=torch.float).unsqueeze(0).to(device)).argmax(1)
    else:
        return random.randint(0, n_actions-1)
    

def compute_target_values(batch_size, reward, next_state, termination_flag):
    # next state reward is 0 in the case of termination
    next_state_values = torch.zeros(batch_size, device=device)

    with torch.no_grad():
        # The Q value corresponds to the argmax action
        next_state_values[~termination_flag] = target_net(next_state).max(1)[0]
    return reward + (DISCOUNT_FACTOR * next_state_values.unsqueeze(1))


def update_model():
    # Sample batch from the dataset of transitions (and a mask for final states)
    state_batch, action_batch, next_state_batch, reward_batch, termination_flag_mask, batch_size = memory.sample(BATCH_SIZE)
    state_batch = state_batch.to(device)
    action_batch = action_batch.to(device)
    next_state_batch = next_state_batch.to(device)
    reward_batch = reward_batch.to(device)
    termination_flag_mask = termination_flag_mask.to(device) # true if terminated, false otherwise

    # Compute Q(s_t, a)
    Q_values = policy_net(state_batch).gather(1, action_batch)

    targets = compute_target_values(batch_size, reward_batch, next_state_batch, termination_flag_mask)

    # Compute loss
    loss = loss_fn(Q_values, targets)

    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

    torch.cuda.empty_cache()
    gc.collect()

    return loss


def update_target():
    if HARD_UPDATE:
        if update_modeling_step % HARD_UPDATE_STEPS == 0:
           # Hard update of the target network's weights
            # θ′ ← θ
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]
            target_net.load_state_dict(target_net_state_dict) 
    else:
        # Soft update of the target network's weights via polyak averaging
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description='Deep Q-Learning Hyperparameters')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--eps_start', type=float, default=0.9, help='Start value of epsilon')
    parser.add_argument('--eps_end', type=float, default=0.05, help='End value of epsilon')
    parser.add_argument('--exp_anneal_samples', type=int, default=400, help='Max number of episodes that exploration probability is annealed over')
    parser.add_argument('--tau', type=float, default=0.005, help='Soft update weight')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--episodes', type=int, default=500, help='Num episodes')
    parser.add_argument('--max_steps', type=int, default=1000, help='Maximum steps per episode')
    parser.add_argument('--offline_training', type=int, default=0, help='Additional offline update_modeling')
    parser.add_argument('--simulation', type=str, default="sim1", help='Simulation json')
    parser.add_argument('--draw_neighbourhood', action="store_true", help='Draw neighbourhood')
    parser.add_argument('--test', action="store_true", help='Test out agent')
    parser.add_argument('--animate', action="store_true", help='Animate (whether testing or not)')
    parser.add_argument('--wandb_project', type=str, help='Save results to wandb in the specified project')
    parser.add_argument('--experiment_name', type=str, help='Name of experiment in wandb')
    parser.add_argument('--model', default='policy_net', type=str, help='Name of model to store/load')
    parser.add_argument('--hard_update', default=0, type=int, help='Number of update_modeling steps between hard update of target network, if <= 0, then do soft updates')

    parser.add_argument('--classifier', default='DQN', type=str, help="The type fo classifier to update_model (DQN, FC)")
    # FC Model stuff
    parser.add_argument("--layers", default=[64, 64, 32, 16], nargs="+", type=int, help="List of integers as dimensions of layer")
    # DQN Model stuff
    parser.add_argument('--n_convs', default=2, type=int, help='Number of convolutional layers in DQN')
    parser.add_argument('--kernel_size', default=3, type=int, help='Kernel size in DQN')
    parser.add_argument('--pool_size', default=2, type=int, help='Pooling size in DQN')
    parser.add_argument('--n_out_channels', default=16, type=int, help='Number of output channels after convolutions')
    parser.add_argument('--n_lins', default=3, type=int, help='Number of linear layers after convolutions')
    # Video stuff
    args, remaining = parser.parse_known_args()
    parser.add_argument('--title',  type=str, default=args.simulation, help='Title for video & heatmap to save (defaults to loaded sim.json)(if --animate)')
    parser.add_argument('--save_freq',  type=int, default=args.episodes/3, help='save animation every ~ number of episodes (if --animate). Defaults to intervals of 1/3* --episodes')
    args = parser.parse_args()

    BATCH_SIZE = args.batch_size
    DISCOUNT_FACTOR = args.gamma
    EPS_START = args.eps_start
    EPS_END = args.eps_end
    EXP_ANNEAL_SAMPLES = args.exp_anneal_samples
    TAU = args.tau
    LR = args.lr
    EPISODES = args.episodes
    MAX_STEPS = args.max_steps
    DRAW_NEIGHBOURHOOD = args.draw_neighbourhood
    TEST = args.test
    ANIMATE = args.animate
    OFFLINE_TRAINING_EPS = args.offline_training
    OFFLINE = False
    HARD_UPDATE = True if args.hard_update > 0 else False
    HARD_UPDATE_STEPS = args.hard_update

    if TEST: # There is no need to do multiple episodes when testing
        EPISODES = 1

    # setup wandb
    if args.wandb_project is not None:
        import wandb
        config = vars(args).copy()
        for k in ['draw_neighbourhood', 'test', 'animate', 'wandb_project']:
            config.pop(k, None)
        wandb.init(project=args.wandb_project, config=config, name=args.experiment_name)
        print("Initialized wandb")

    sim = Simulator(f"./simulations/{args.simulation}.json")
    sim.start()
    state_shape, n_actions = sim.info()

    print("Initialized simulator")


    policy_net = DQN(state_shape, n_actions, n_convs=args.n_convs, kernel_size=args.kernel_size, 
                    pool_size=args.pool_size, n_out_channels=args.n_out_channels, n_lins=args.n_lins).to(device)
    target_net = DQN(state_shape, n_actions, n_convs=args.n_convs, kernel_size=args.kernel_size, 
                    pool_size=args.pool_size, n_out_channels=args.n_out_channels, n_lins=args.n_lins).to(device)

    
    if os.path.isfile(f"./models/{args.model}.pth"):
        load_model(policy_net, f"./models/{args.model}.pth", device)

    target_net.load_state_dict(policy_net.state_dict())

    print("Initialized model")

    loss_fn = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayMemory(capacity=5000)

    update_modeling_step = 1
    objective_proportion = 0
    for i_episode in range(EPISODES + OFFLINE_TRAINING_EPS):
        # Empty GPU
        torch.cuda.empty_cache()
        gc.collect()
        # Initialize simulation
        state = sim.start()
        # Initialize animation
        if TEST or ANIMATE:
            anim_frames = [sim.get_current_frame()]
        print(f"Starting{' (offline) ' if OFFLINE else ' '} Episode: {i_episode+1}.")
        # Episodic metrics
        mean_loss = 0
        total_reward = 0
        number_steps = 0
        # Run simulation and update_modeling
        for t in range(MAX_STEPS):
            action = select_action(state)
            next_state, reward, termination_condition = sim.step(action)
            terminated = True if termination_condition != 0 else False
            # Update episodic metrics
            total_reward += reward
            number_steps += 1

            # Store the transition in memory
            if not OFFLINE:
                memory.push(state, action, next_state, reward, terminated)

            # Move to the next state
            state = next_state

            if TEST or ANIMATE:
                # Animate
                anim_frames.append(sim.get_current_frame())
            # if not TEST:
            #     loss = update_model() 
                
            #     # TODO: check this
            #     mean_loss += (loss - mean_loss) / (t + 1)
                
            #     update_target()

            # Increment step
            update_modeling_step += 1
            if (t + 1) % 100 == 0:
                print("Step:", t+1)
            # Check for termination
            if terminated:
                print("Finished at:", t+1)
                # Update objective proportion
                objective_proportion += (1 if termination_condition == 2 else 0 - objective_proportion) / (i_episode + 1)
                break
        if not TEST:
            loss = update_model() 
            
            # TODO: check this
            mean_loss += (loss - mean_loss) / (t + 1)
            
            update_target()
        
        if i_episode < 10 or i_episode % 20 == 0:
            draw_heatmap(sim, args.title)   
        
        # Switch to offline update_modeling
        if not OFFLINE and i_episode >= EPISODES:
            OFFLINE = True

        # Record output
        if args.wandb_project is not None:
            wandb.log({"loss": mean_loss, "objective_proportion": objective_proportion, "reward": total_reward, 
                       "number_steps": number_steps, "episode": i_episode, "step": update_modeling_step})
        # Save model every 100 episodes
        if (i_episode + 1) % 100 == 0 and not TEST:
            save_model(policy_net, f"./models/{args.model}.pth")
        
        if TEST or ANIMATE:
            SimAnimation(sim.bodies, sim.objective, sim.limits, anim_frames, len(anim_frames), i_episode + 1, args.save_freq, args.title, 
                         DRAW_NEIGHBOURHOOD, sim.grid_radius, sim.box_width)

    # Save final model
    if not TEST:
        save_model(policy_net, f"./models/{args.model}.pth")

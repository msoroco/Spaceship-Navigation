import os
import time
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

from PPO import ActorCritic


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

    # plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust layout to make space for colorbar
    plt.show()

    plt.show()
    plt.savefig('heatmap_' + title +'_DoubleDQN.png')


def select_action(state):
    """ the exploration probability starts with a value of `EPS_START` and
        then linearly anneals to `EPS_END` over the course of `EXP_ANNEAL_SAMPLES`
        timesteps."""
    if len(state.shape) == 3:
        state = np.expand_dims(state, axis=0)
    with torch.no_grad():
        state = torch.FloatTensor(state).to(device)
        action, action_logprob, state_val = old_policy_net.act(state)
    return action, action_logprob, state_val


def update_model(MEMORY):
    # sequence: state, action, next_state, reward, terminated, action, action_logprob, state_val
    rewards = []
    discounted_reward = 0
    for m in reversed(MEMORY):
        reward, is_terminal = m[3], m[4]
        if is_terminal:
            discounted_reward = 0
        discounted_reward = reward + (GAMMA * discounted_reward)
        rewards.insert(0, discounted_reward)
        
    # Normalizing the rewards
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
    if len(rewards) > 1:
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

    # convert list to tensor
    old_states = torch.squeeze(torch.stack([torch.from_numpy(m[0]) for m in MEMORY], dim=0)).detach().to(device)
    old_actions = torch.squeeze(torch.stack([m[1] for m in MEMORY], dim=0)).detach().to(device)
    old_logprobs = torch.squeeze(torch.stack([m[-2] for m in MEMORY], dim=0)).detach().to(device)
    old_state_values = torch.squeeze(torch.stack([m[-1] for m in MEMORY], dim=0)).detach().to(device)

    # calculate advantages
    advantages = rewards.detach() - old_state_values.detach()

    wandb_loss = 0
    if len(MEMORY) == 1:
        old_states = old_states.unsqueeze(0)
        old_actions = old_actions.unsqueeze(0)
        old_logprobs = old_logprobs.unsqueeze(0)


    for _ in range(K_EPOCHS):

        # Evaluating old actions and values
        logprobs, state_values, dist_entropy = policy_net.evaluate(old_states.float(), old_actions.float())

        # match state_values tensor dimensions with rewards tensor
        state_values = torch.squeeze(state_values)
        
        # Finding the ratio (pi_theta / pi_theta__old)
        ratios = torch.exp(logprobs - old_logprobs.detach())

        # Finding Surrogate Loss  
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1-EPS_CLIP, 1+EPS_CLIP) * advantages

        # final loss of clipped objective PPO
        loss = -torch.min(surr1, surr2) + 0.5 * loss_fn(state_values, rewards) - 0.01 * dist_entropy
        
        # take gradient step
        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()

        wandb_loss += loss.mean().item()
            
    # Copy new weights into old policy
    old_policy_net.load_state_dict(policy_net.state_dict())

    torch.cuda.empty_cache()
    gc.collect()

    return wandb_loss/K_EPOCHS


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
    parser.add_argument('--episodes', type=int, default=3000, help='Num episodes')
    parser.add_argument('--start_episode', type=int, default=0, help='Start episode')
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
    START_EPISODE = args.start_episode
    MAX_STEPS = args.max_steps
    DRAW_NEIGHBOURHOOD = args.draw_neighbourhood
    TEST = args.test
    ANIMATE = args.animate
    OFFLINE_TRAINING_EPS = args.offline_training
    OFFLINE = False
    HARD_UPDATE = True if args.hard_update > 0 else False
    HARD_UPDATE_STEPS = args.hard_update
    GAMMA = 0.99
    K_EPOCHS = 80
    EPS_CLIP = 0.2

    if TEST and ANIMATE: # There is no need to do multiple episodes when animating a test run
        EPISODES = 1
        OFFLINE_TRAINING_EPS = 0
    if TEST:
        EPISODES = 100
        OFFLINE_TRAINING_EPS = 0

    # setup wandb
    if args.wandb_project is not None:
        import wandb
        run_id_file = f"./runid_{args.experiment_name}.txt"
        if os.path.exists(run_id_file):
            with open(run_id_file, 'r') as f:
                run_id = f.read().strip()
        else:
            run_id = wandb.util.generate_id()
            with open(run_id_file, 'w') as f:
                f.write(run_id)

        # in case run_id is empty (first run)
        if not run_id:
            run_id = wandb.util.generate_id()
            with open(run_id_file, 'w') as f:
                f.write(run_id)

        config = vars(args).copy()
        for k in ['draw_neighbourhood', 'test', 'animate', 'wandb_project']:
            config.pop(k, None)
        wandb.init(project=args.wandb_project, config=config, name=args.experiment_name, id=run_id, resume="allow")
        print("Initialized wandb")

    sim = Simulator(f"./simulations/{args.simulation}.json")
    sim.start()
    state_shape, n_actions = sim.info()

    # print("Initialized simulator")

    policy_net     = ActorCritic(np.prod(state_shape), n_actions).to(device)
    old_policy_net = ActorCritic(np.prod(state_shape), n_actions).to(device)
    old_policy_net.load_state_dict(policy_net.state_dict())
    
    if os.path.isfile(f"./models/{args.model}.pth"):
        if os.path.isfile(f"./models/{args.model}_epis=5999.pth"):
            load_model(policy_net, f"./models/{args.model}_epis=5999.pth", device)
        else:
            load_model(policy_net, f"./models/{args.model}.pth", device)
        # print('here')
        if TEST:
            EPISODES = 100
            OFFLINE_TRAINING_EPS = 0
        else:
            START_EPISODE = 6000
            EPISODES = 15000

    # print("Initialized model")

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam([
                        {'params': policy_net.actor.parameters(), 'lr': LR},
                        {'params': policy_net.critic.parameters(), 'lr': LR*5}
                    ])
    memory = ReplayMemory(capacity=5000)

    if not TEST:
        training_step = 1
    else:
        test_return = 0
        test_episodes_length = 0
        test_returns = 0
        test_episodes_lengths = 0
    objective_proportion = 0

    for i_episode in range(START_EPISODE, EPISODES + OFFLINE_TRAINING_EPS):
        # Empty GPU
        torch.cuda.empty_cache()
        gc.collect()
        # Initialize simulation
        state = sim.start()
        # Initialize animation
        if ANIMATE:
            anim_frames = [sim.get_current_frame()]
        # print(f"Starting{' (offline) ' if OFFLINE else ' '} Episode: {i_episode+1}.")
        # Episodic metrics
        mean_loss = 0
        total_reward = 0
        number_steps = 0
        # Run simulation and update_modeling
        start = time.time()
        MEMORY = []
        for t in range(MAX_STEPS):
            action, action_logprob, state_val = select_action(state)
            next_state, reward, termination_condition = sim.step(action)
            terminated = True if termination_condition != 0 else False
            # Update episodic metrics
            total_reward += reward
            number_steps += 1

            # Store the transition in memory
            if not OFFLINE:
                MEMORY.append([state, action, next_state, reward, terminated, action, action_logprob, state_val])

            # Move to the next state
            state = next_state

            if ANIMATE:
                # Animate
                anim_frames.append(sim.get_current_frame())
            
            # Check for termination
            if terminated:
                # Update objective proportion
                objective_proportion += ((1 if termination_condition == 2 else 0) - objective_proportion) / (i_episode + 1)
                break
        if not TEST:
            loss = update_model(MEMORY) 
            
            # TODO: check this
            mean_loss += (loss - mean_loss) / (t + 1)

            # Increment step
            training_step += 1
            # if (i_episode) % 10 == 0:
            #     print(f"Episode {i_episode+1}, Wall_Time: {time.time() - start}, Training_step: {training_step}, Step_Loss: {loss}, Train_episode_length {t+1}, Mean_Loss: {mean_loss}, objective_reached_rate: {objective_proportion}, offline_learning: {OFFLINE}")
        else:
            test_return = total_reward
            test_episodes_length = number_steps
            test_returns += test_return
            # print(f"Episode {i_episode+1}, Test_return: {test_return}, Test_Episode_Length: {test_episodes_length}, Wall_Time: {time.time() - start}")
        
        if i_episode < 10 or i_episode % 20 == 0:
            draw_heatmap(sim, args.title)   
        
        # Switch to offline update_modeling
        if not OFFLINE and i_episode >= EPISODES:
            OFFLINE = True

        # Record output
        if args.wandb_project is not None:
            if not TEST:
                wandb.log({
                    "Training/loss": mean_loss,
                    "Training/objective_proportion": objective_proportion,
                    "Training/reward": total_reward,
                    "Training/number_steps": number_steps,
                    "Training/episode": i_episode,
                    "Training/step": training_step
                })
            else:
                wandb.log({
                    "Testing/test_reward": test_return,
                    "Testing/test_episodes_length": test_episodes_length,
                    "Testing/test_episode": i_episode
                })
                test_returns += min(test_return, 0)
        # Save model every 100 episodes
        if (i_episode + 1) % 100 == 0 and not TEST:
            save_model(policy_net, f"./models/{args.model}.pth")
        
        if ANIMATE:
            if TEST:
                SimAnimation(sim.bodies, sim.objective, sim.limits, anim_frames, len(anim_frames), i_episode + 1, save_freq=1, title=args.title + "_DoubleDQN", 
                    draw_neighbourhood=DRAW_NEIGHBOURHOOD, grid_radius=sim.grid_radius, box_width=sim.box_width)
            else:
                SimAnimation(sim.bodies, sim.objective, sim.limits, anim_frames, len(anim_frames), i_episode + 1, args.save_freq, args.title + "_DoubleDQN", 
                            DRAW_NEIGHBOURHOOD, sim.grid_radius, sim.box_width)
            
    
    print({
        "Model_path": args.model,
        "Simulation": args.simulation,
        "Testing/avg_test_reward": test_returns/i_episode,
        "Objective_Proportion": objective_proportion
    })
    # Save final model
    if not TEST:
        # checkpoint
        print(f"simulation: {args.simulation}, model: {args.model}, episodes: {EPISODES}")
        model_path = f"./models/{args.model}_epis={i_episode}.pth"
        save_model(policy_net, model_path)
        wandb.save(model_path)
        # update model
        save_model(policy_net, f"./models/{args.model}.pth")
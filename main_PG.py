import torch
import torch.optim as optim
from simulator import Simulator
from model import PolicyNetwork
from replay import ReplayMemory
import numpy as np

# Hyperparameters
num_episodes = 1000
gamma = 0.99  # Discount factor
learning_rate = 0.01
buffer_capacity = 10000  # Capacity of replay buffer
batch_size = 32

# Initialize environment and policy network
env = Simulator(filepath="simulations/sim2.json")
state = env.start()  # Start the environment
state_dim = np.prod(state.shape)  # Flattened size of the state (10 * 21 * 21 = 4410)
action_dim = env.info()[1]  # Get number of actions
policy = PolicyNetwork(input_dim=state_dim, output_dim=action_dim)
optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

# Initialize replay buffer
replay_buffer = ReplayMemory(buffer_capacity)

# Training loop
for episode in range(num_episodes):
    state = torch.tensor(env.start(), dtype=torch.float32)
    log_probs = []
    rewards = []

    done = False
    while not done:
        # Get action probabilities
        action_probs = policy(state.flatten())
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()

        # Step in the environment
        next_state, reward, termination_condition = env.step(action.item())
        done = termination_condition != 0  # Check if episode is over

        # Store log probabilities and rewards
        log_probs.append(action_dist.log_prob(action))
        rewards.append(reward)

        # Store the transition in the replay buffer
        replay_buffer.push(state.numpy(), action.item(), next_state, reward, done)

        state = torch.tensor(next_state, dtype=torch.float32)

    # Compute discounted rewards-to-go
    R = 0
    discounted_rewards = []
    for r in rewards[::-1]:
        R = r + gamma * R
        discounted_rewards.insert(0, R)

    # Normalize rewards for stability
    discounted_rewards = torch.tensor(discounted_rewards)
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

    # Policy Gradient update
    optimizer.zero_grad()
    policy_loss = []
    for log_prob, R in zip(log_probs, discounted_rewards):
        policy_loss.append(-log_prob * R)
    policy_loss = torch.stack(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()

    print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {sum(rewards)}")

    # Optionally sample from replay buffer (e.g., for experience replay)
    if len(replay_buffer) > batch_size:
        state_batch, action_batch, next_state_batch, reward_batch, final_state_mask, _ = replay_buffer.sample(batch_size)
        # Further training steps using the sampled batch, such as Q-value updates, could go here

# Save the trained policy
torch.save(policy.state_dict(), "policy_network.pth")

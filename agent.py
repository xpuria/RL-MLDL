import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal


def discount_rewards(r, gamma):
    discounted_r = torch.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size(-1))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.tanh = torch.nn.Tanh()

        """
            Actor network
        """
        self.fc1_actor = torch.nn.Linear(state_space, self.hidden)
        self.fc2_actor = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_actor_mean = torch.nn.Linear(self.hidden, action_space)
        
        # Learned standard deviation for exploration at training time 
        self.sigma_activation = F.softplus
        init_sigma = 0.5
        self.sigma = torch.nn.Parameter(torch.zeros(self.action_space)+init_sigma)


        """
            Critic network
        """
        self.fc1_critic = torch.nn.Linear(state_space, self.hidden)
        self.fc2_critic = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_critic = torch.nn.Linear(self.hidden, 1)


        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)


    def forward(self, x):
        """
            Actor
        """
        x_actor = self.tanh(self.fc1_actor(x))
        x_actor = self.tanh(self.fc2_actor(x_actor))
        action_mean = self.fc3_actor_mean(x_actor)

        sigma = self.sigma_activation(self.sigma)
        normal_dist = Normal(action_mean, sigma)


        """
            Critic
        """
        x_critic = self.tanh(self.fc1_critic(x))
        x_critic = self.tanh(self.fc2_critic(x_critic))
        state_value = self.fc3_critic(x_critic)

        
        return normal_dist, state_value


class Agent(object):
    def __init__(self, policy, device='cpu'):
        self.train_device = device
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

        self.gamma = 0.99
        self.baseline = 20.0
        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []


    def update_policy(self, algorithm):
        action_log_probs = torch.stack(self.action_log_probs, dim=0).to(self.train_device).squeeze(-1)
        states = torch.stack(self.states, dim=0).to(self.train_device).squeeze(-1)
        next_states = torch.stack(self.next_states, dim=0).to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        done = torch.Tensor(self.done).to(self.train_device)

        discounted_returns = discount_rewards(rewards, self.gamma)

        if algorithm == 'reinforce':                    
            # Normalize returns
            returns = (discounted_returns - discounted_returns.mean()) / (discounted_returns.std() + 1e-9)
            # Compute policy gradient loss
            policy_loss = -torch.sum(action_log_probs * returns)
        
        elif algorithm == 'reinforce_baseline':
            
            # Subtract the baseline
            adjusted_returns = discounted_returns - self.baseline

            # Normalize the adjusted returns
            adjusted_returns = (adjusted_returns - adjusted_returns.mean()) / (adjusted_returns.std() + 1e-8)

            # Compute policy gradient loss
            policy_loss = -torch.sum(action_log_probs * adjusted_returns)

        elif algorithm == 'actor_critic':
            # Compute advantages
            state_values = torch.stack([self.policy(s)[1].squeeze() for s in states])
            next_state_values = torch.stack([self.policy(ns)[1].squeeze() for ns in next_states])
            td_targets = rewards + self.gamma * next_state_values * (1 - done)
            td_targets = td_targets.unsqueeze(1)  # Reshape to [batch_size, 1] if needed
            state_values = state_values.unsqueeze(1)  # Reshape to [batch_size, 1] if needed
            advantages = td_targets - state_values

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Policy loss
            actor_loss = -torch.sum(action_log_probs * advantages)

            # Critic loss
            critic_loss = F.mse_loss(state_values, td_targets)

            # Total loss
            policy_loss = actor_loss + critic_loss

        # Backpropagation and optimization step
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        # Clear the trajectory data
        self.states, self.next_states, self.action_log_probs, self.rewards, self.done = [], [], [], [], []
        

    def get_action(self, state, evaluation=False):
        """ state -> action (3-d), action_log_densities """
        x = torch.from_numpy(state).float().to(self.train_device)

        normal_dist, state_value = self.policy(x)

        if evaluation:  # Return mean
            return normal_dist.mean, None

        else:   # Sample from the distribution
            action = normal_dist.sample()

            # Compute Log probability of the action [ log(p(a[0] AND a[1] AND a[2])) = log(p(a[0])*p(a[1])*p(a[2])) = log(p(a[0])) + log(p(a[1])) + log(p(a[2])) ]
            action_log_prob = normal_dist.log_prob(action).sum()

            return action, action_log_prob


    def store_outcome(self, state, next_state, action_log_prob, reward, done):
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.done.append(done)


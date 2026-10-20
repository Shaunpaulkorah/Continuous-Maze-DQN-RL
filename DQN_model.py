# Import:
# -------
import random
import torch.nn as nn  # learnable parameters with trainable weights
import torch.nn.functional as F # pure mathematical functions (no trainable weights) like ReLU


# Deep Q-Network:
# ---------------
class Qnet(nn.Module):
    def __init__(self, dim_actions, dim_states):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(dim_states, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, dim_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
    
    def sample_action(self, observation, epsilon):
        a = self.forward(observation)
        
        #! Exploration
        if random.random() < epsilon:
            return random.randint(0, 3)   # 4 actions: up, down, left, right
        
        #! Exploitation
        else : 
            return a.argmax().item()

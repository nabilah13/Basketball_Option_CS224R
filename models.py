import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN_model(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.layer1 = nn.Linear(num_features, 15)
        self.layer2 = nn.Linear(15, 15)
        self.layer3 = nn.Linear(15, 1)

        # Initialize the linear layers
        nn.init.uniform_(self.layer1.weight, a = -0.1, b = 0.1)
        nn.init.uniform_(self.layer2.weight, a = -0.1, b = 0.1)
        nn.init.uniform_(self.layer3.weight, a = -0.1, b = 0.1)
        nn.init.zeros_(self.layer1.bias)
        nn.init.zeros_(self.layer2.bias)
        nn.init.zeros_(self.layer3.bias)
        
    def forward(self, x):
        x2 = F.softplus(self.layer1(x))
        x3 = F.softplus(self.layer2(x2))
        return self.layer3(x3)
    
class LSPI_model(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.layer1 = nn.Linear(num_features, 1, bias=True)
        # Initialize the linear layer
        nn.init.uniform_(self.layer1.weight, a = -0.1, b = 0.1)
        nn.init.zeros_(self.layer1.bias)

    def forward(self, x):
        x2 = self.layer1(x)
        return x2
    
class LSPI_model_np():
    def __init__(self, num_features: int):
        self.A_inv = np.eye(num_features)
        self.b = np.zeros(num_features)
        self.wts = np.zeros(num_features).reshape(-1,1)

    def forward(self, x):
        x = x.numpy().reshape(1,-1)
        return x.dot(self.wts)

    def update(self, state, reward, next_state, next_reward, terminal, next_terminal):
        # Update parameters using Sherman Morrison incremental inverse
        q_value_next = self.forward(next_state.numpy(), self.wts)
        I_c1 = (1-1.0*next_terminal) * (q_value_next > next_reward)
        I_c2 = 1.0 - I_c1

        phi1 = state.numpy().reshape(-1,1)
        phi2 =  phi1 - I_c1 * next_state.numpy().reshape(-1,1)
        temp = self.A_inv.T.dot(phi2)
        self.A_inv -= np.outer(self.A_inv.dot(phi1), temp) / (1 + phi1.dot(temp))
        self.b += phi1 * I_c2 * next_reward

        self.wts = self.A_inv.dot(self.b)

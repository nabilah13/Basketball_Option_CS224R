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
    
# class LSPI_model(nn.Module):
#     def __init__(self, num_features: int):
#         super().__init__()
#         # Create parameters for incremental inverse computation
#         self.w = nn.Parameter(torch.zeros(num_features, 1))
#         self.A_inv = nn.Parameter(torch.zeros(num_features, num_features))
#         self.b = nn.Parameter(torch.zeros(num_features, 1))

#     def update_inverse(self, state, reward, next_state, terminal):    
#         # create two indicator variables to aid in calculation
#         i_2 = 1.0 * (terminal | (reward > next_state @ self.w))
#         # set i_1 to 1.0 if i_2 is 0.0, otherwise set it to 0.0
#         i_1 = 1.0 - i_2

#         #Use Sherman-Morrison incremental inverse, updates on a single sample
#         self.A_inv = self.A_inv - (self.A_inv @ x @ x.T @ self.A_inv)/(1 + x.T @ self.A_inv @ x)

#         i_1
#         i_2
#         self.b = self.b + i_2 * ()
#         self.w = self.A_inv @ self.b

#         return

#     def forward(self, state):
#         return state @ self.w
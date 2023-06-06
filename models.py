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

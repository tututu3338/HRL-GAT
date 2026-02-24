import torch
import torch.nn as nn


class ActorNetwork(nn.Module):
    """Actor network that scores each candidate node given composite action features.
    
    Paper Eq.13: f_i^(t) = [z_i || z̄_St || δ_i^(t) || ψ_i]
    Output: scalar score per candidate node.
    """
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        return self.net(x).squeeze(-1)


class CriticNetwork(nn.Module):
    """Critic network that estimates state value V(s_t).
    
    State is represented by the mean embedding of current seed set.
    """
    def __init__(self, input_dim, hidden_dim=128):
        super(CriticNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x):
        return self.net(x)
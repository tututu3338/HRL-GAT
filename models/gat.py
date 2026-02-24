import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GATEncoder(nn.Module):
    """Multi-layer GAT encoder with residual connections and batch normalization.
    
    Paper Section 4.2: Node Embedding with GAT (Eq.3-9)
    """
    def __init__(self, in_dim, hidden_dim, out_dim, device, heads=8):
        super(GATEncoder, self).__init__()
        self.device = device
        
        self.gat1 = GATConv(in_dim, hidden_dim // heads, heads=heads).to(self.device)
        self.gat2 = GATConv(hidden_dim, hidden_dim // heads, heads=heads).to(self.device)
        self.gat3 = GATConv(hidden_dim, out_dim, heads=1, concat=False).to(self.device)
        
        # Residual connections (Paper Eq.8)
        self.res1 = nn.Linear(in_dim, hidden_dim).to(self.device)
        self.res2 = nn.Linear(hidden_dim, hidden_dim).to(self.device)
        self.res3 = nn.Linear(hidden_dim, out_dim).to(self.device)
        
        self.bn1 = nn.BatchNorm1d(hidden_dim).to(self.device)
        self.bn2 = nn.BatchNorm1d(hidden_dim).to(self.device)
        
        try:
            for layer in [self.gat1, self.gat2, self.gat3]:
                if hasattr(layer, 'lin'):
                    nn.init.xavier_normal_(layer.lin.weight)
                    if layer.lin.bias is not None:
                        nn.init.zeros_(layer.lin.bias)
            for layer in [self.res1, self.res2, self.res3]:
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
        except Exception as e:
            print(f"Weight initialization warning: {e}")

    def forward(self, x, edge_index):
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        
        # Layer 1 (Paper Eq.9)
        h_gat = self.gat1(x, edge_index)
        h_res = self.res1(x)
        h1 = self.bn1(h_gat + h_res)
        h1 = F.relu(F.dropout(h1, p=0.1, training=self.training))
        
        # Layer 2
        h_gat = self.gat2(h1, edge_index)
        h_res = self.res2(h1)
        h2 = self.bn2(h_gat + h_res)
        h2 = F.relu(F.dropout(h2, p=0.1, training=self.training))
        
        # Layer 3
        out = self.gat3(h2, edge_index) + self.res3(h2)
        return out

    def pretrain(self, features, edge_index, epochs, lambda_smooth=0.1):
        """Contrastive + smoothing pretraining (Paper Eq.12).
        
        L_pre = L_con + λ_s · L_smooth
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.train()
        
        # Precompute node degrees from edge_index for smoothing loss weights
        src, dst = edge_index[0], edge_index[1]
        all_nodes = torch.cat([src, dst])
        node_degrees = torch.zeros(features.size(0), device=self.device)
        node_degrees.scatter_add_(0, all_nodes, torch.ones_like(all_nodes, dtype=torch.float))
        node_degrees = node_degrees.clamp(min=1)
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            encoded = self(features, edge_index)
            
            # Sample edges for contrastive loss
            num_edges = min(10000, edge_index.size(1))
            edge_idx = torch.randperm(edge_index.size(1))[:num_edges]
            pos_edges = edge_index[:, edge_idx]
            
            neg_edges = torch.randint(0, features.size(0), (2, num_edges), device=self.device)
            
            # Contrastive loss L_con
            pos_score = torch.sum(encoded[pos_edges[0]] * encoded[pos_edges[1]], dim=1)
            neg_score = torch.sum(encoded[neg_edges[0]] * encoded[neg_edges[1]], dim=1)
            
            labels = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)]).to(self.device)
            scores = torch.cat([pos_score, neg_score])
            loss_con = F.binary_cross_entropy_with_logits(scores, labels)
            
            # Smoothing loss L_smooth (Paper Eq.11)
            # L_smooth = (1/|E|) Σ_{(i,j)∈E} w_ij · ||z_i - z_j||²
            # where w_ij = p_ij = 1/d_j (WIC edge probability)
            z_src = encoded[pos_edges[0]]
            z_dst = encoded[pos_edges[1]]
            z_diff = z_src - z_dst
            sq_diff = (z_diff ** 2).sum(dim=1)
            edge_weights = 1.0 / node_degrees[pos_edges[1]]
            loss_smooth = (edge_weights * sq_diff).mean()
            
            # Combined loss (Paper Eq.12)
            loss = loss_con + lambda_smooth * loss_smooth
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            optimizer.step()
            
            if (epoch + 1) % 20 == 0:
                print(f"Pretrain Epoch {epoch+1}/{epochs}, "
                      f"L_con: {loss_con.item():.4f}, "
                      f"L_smooth: {loss_smooth.item():.4f}, "
                      f"Total: {loss.item():.4f}")
        
        self.eval()
import torch
import torch.nn.functional as F
import numpy as np
import os
from models.policy import ActorNetwork, CriticNetwork


class PPOAgent:
    """PPO Agent for sequential seed selection
    
    Implements:
    - Composite action features f_i = [z_i || z̄_St || δ_i || ψ_i]
    - PPO clipped objective 
    - GAE advantage estimation
    - Total loss with entropy regularization
    """
    def __init__(self, embedding_dim, n_nodes, device, cfg):
        self.device = device
        self.n_nodes = n_nodes
        self.embedding_dim = embedding_dim
        
        # Composite feature dim: z_i(d) + z̄_St(d) + δ_i(1) + ψ_i(3) = 2d + 4
        self.action_feature_dim = 2 * embedding_dim + 4
        # State dim for critic: z̄_St(d)
        self.state_dim = embedding_dim
        
        self.actor = ActorNetwork(self.action_feature_dim).to(device)
        self.critic = CriticNetwork(self.state_dim).to(device)
        
        self.optimizer = torch.optim.Adam([
            {'params': self.actor.parameters()},
            {'params': self.critic.parameters()}
        ], lr=cfg.lr)
        
        self.clip_ratio = cfg.clip_ratio
        self.max_grad_norm = cfg.max_grad_norm
        self.temperature = cfg.temperature
        self.gamma = cfg.gamma
        self.gae_lambda = cfg.gae_lambda
        self.entropy_coef = getattr(cfg, 'entropy_coef', 0.01)
        self.value_coef = getattr(cfg, 'value_coef', 0.5)
        self.ppo_epochs = getattr(cfg, 'ppo_epochs', 4)

    def build_composite_features(self, node_embeddings, valid_nodes, seed_set, env):
        """Build composite action features.
        
        f_i^(t) = [z_i || z̄_St || δ_i^(t) || ψ_i]
        
        Args:
            node_embeddings: (n_nodes, d) tensor from GAT encoder
            valid_nodes: list of candidate node indices
            seed_set: list of already selected seed indices
            env: InfluenceEnvironment instance
            
        Returns:
            composite: (n_valid, 2d+4) tensor
        """
        valid_nodes_t = torch.tensor(valid_nodes, dtype=torch.long, device=self.device)
        z_i = node_embeddings[valid_nodes_t]  # (n_valid, d)
        n_valid = len(valid_nodes)
        
        # z̄_St: mean embedding of current seed set
        if seed_set:
            seed_indices = torch.tensor(seed_set, dtype=torch.long, device=self.device)
            seed_embeddings = node_embeddings[seed_indices]  # (|St|, d)
            z_bar = seed_embeddings.mean(dim=0, keepdim=True).expand(n_valid, -1)  # (n_valid, d)
            
            # δ_i^(t): diversity feature = 1 - max_sim(z_i, z_v∈St)
            z_i_norm = F.normalize(z_i, dim=1)
            seed_norm = F.normalize(seed_embeddings, dim=1)
            sim_matrix = torch.mm(z_i_norm, seed_norm.t())  # (n_valid, |St|)
            max_sim = sim_matrix.max(dim=1)[0]  # (n_valid,)
            delta_i = (1 - max_sim).unsqueeze(1)  # (n_valid, 1)
        else:
            z_bar = torch.zeros(n_valid, self.embedding_dim, device=self.device)
            delta_i = torch.ones(n_valid, 1, device=self.device)
        
        # ψ_i: static structural features [degree_norm, clustering_coef, ecmr_norm]
        psi_list = [env.get_structural_features(node) for node in valid_nodes]
        psi_i = torch.tensor(psi_list, dtype=torch.float32, device=self.device)  # (n_valid, 3)
        
        # Concatenate: [z_i || z̄_St || δ_i || ψ_i]
        composite = torch.cat([z_i, z_bar, delta_i, psi_i], dim=1)  # (n_valid, 2d+4)
        return composite

    def get_state_repr(self, node_embeddings, seed_set):
        """Get state representation for critic: mean embedding of current seeds."""
        if seed_set:
            seed_indices = torch.tensor(seed_set, dtype=torch.long, device=self.device)
            state = node_embeddings[seed_indices].mean(dim=0)
        else:
            state = torch.zeros(self.embedding_dim, device=self.device)
        return state

    def get_action(self, node_embeddings, valid_nodes, seed_set, env, deterministic=False):
        """Select an action using composite features and actor scoring.
        
        Args:
            node_embeddings: (n_nodes, d) from GAT
            valid_nodes: list of candidate node indices
            seed_set: list of already selected seeds
            env: InfluenceEnvironment
            deterministic: if True, use argmax instead of sampling
            
        Returns:
            (selected_node, log_prob, action_idx) or (None, None, None)
        """
        if not valid_nodes:
            return None, None, None
            
        with torch.no_grad():
            composite_features = self.build_composite_features(
                node_embeddings, valid_nodes, seed_set, env
            )
            scores = self.actor(composite_features) / self.temperature  # (n_valid,)
            probs = F.softmax(scores, dim=0)
            
            if not torch.isfinite(probs).all():
                return None, None, None
            
            if deterministic:
                action_idx = torch.argmax(probs)
            else:
                dist = torch.distributions.Categorical(probs)
                action_idx = dist.sample()
            
            dist = torch.distributions.Categorical(probs)
            log_prob = dist.log_prob(action_idx)
            
            selected_node = valid_nodes[action_idx.item()]
            return selected_node, log_prob, action_idx

    def compute_gae(self, values, rewards):
        """Compute GAE advantages and target returns .
        
        Returns = Advantages + Values (Paper: R̂_t = Â_t + V_φ(s_t))
        """
        advantages = torch.zeros_like(rewards)
        last_gae = 0.0
        for t in reversed(range(len(rewards))):
            next_value = values[t + 1] if t < len(rewards) - 1 else 0.0
            delta = rewards[t] + self.gamma * next_value - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * last_gae
            advantages[t] = last_gae
        returns = advantages + values  # Paper: R̂_t = Â_t + V_φ(s_t)
        return advantages, returns

    def update(self, trajectory):
        """PPO update with proper clipping, entropy bonus, consistent probabilities.
        
       
        
        Args:
            trajectory: list of dicts with keys:
                - candidate_features: (n_valid, feat_dim) tensor
                - action_idx: tensor, index into candidates
                - old_log_prob: float
                - reward: float
                - state_repr: (d,) tensor
        """
        if len(trajectory) < 2:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        
        rewards = torch.tensor(
            [t['reward'] for t in trajectory], dtype=torch.float32, device=self.device
        )
        old_log_probs = torch.tensor(
            [t['old_log_prob'] for t in trajectory], dtype=torch.float32, device=self.device
        )
        state_reprs = torch.stack([t['state_repr'] for t in trajectory])  # (T, d)
        
        # Compute GAE with current critic values
        with torch.no_grad():
            values = self.critic(state_reprs).squeeze()  # (T,)
            advantages, returns = self.compute_gae(values, rewards)
            if len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO multi-epoch update
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        
        for _ in range(self.ppo_epochs):
            epoch_policy_loss = torch.tensor(0.0, device=self.device)
            epoch_value_loss = torch.tensor(0.0, device=self.device)
            epoch_entropy = torch.tensor(0.0, device=self.device)
            
            for t, step_data in enumerate(trajectory):
                candidate_features = step_data['candidate_features']  # (n_valid, feat_dim)
                action_idx = step_data['action_idx']
                
                if not isinstance(action_idx, torch.Tensor):
                    action_idx = torch.tensor(action_idx, device=self.device)
                
                # Recompute action distribution with current actor parameters
                scores = self.actor(candidate_features) / self.temperature
                probs = F.softmax(scores, dim=0)
                dist = torch.distributions.Categorical(probs)
                
                new_log_prob = dist.log_prob(action_idx)
                entropy = dist.entropy()
                
                # PPO clipped objective 
                ratio = torch.exp(new_log_prob - old_log_probs[t])
                surr1 = ratio * advantages[t]
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages[t]
                policy_loss = -torch.min(surr1, surr2)
                
                # Critic loss 
                value_pred = self.critic(state_reprs[t].unsqueeze(0)).squeeze()
                value_loss = F.mse_loss(value_pred, returns[t])
                
                epoch_policy_loss = epoch_policy_loss + policy_loss
                epoch_value_loss = epoch_value_loss + value_loss
                epoch_entropy = epoch_entropy + entropy
            
            n_steps = len(trajectory)
            # Total loss (Paper Eq.23): L = -L^PPO + β·L^critic - η·H(π)
            loss = (epoch_policy_loss / n_steps + 
                    self.value_coef * epoch_value_loss / n_steps - 
                    self.entropy_coef * epoch_entropy / n_steps)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            total_policy_loss += epoch_policy_loss.item() / n_steps
            total_value_loss += epoch_value_loss.item() / n_steps
            total_entropy += epoch_entropy.item() / n_steps
        
        return {
            "policy_loss": total_policy_loss / self.ppo_epochs,
            "value_loss": total_value_loss / self.ppo_epochs,
            "entropy": total_entropy / self.ppo_epochs
        }

    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }, path)

    def load(self, path):
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic.load_state_dict(checkpoint['critic'])
            print(f"PPO model loaded: {path}")
        else:
            print(f"Model file not found: {path}")

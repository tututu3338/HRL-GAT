import os
import torch
import torch.nn.functional as F
import numpy as np
import random
import time
from tqdm import tqdm

from utils.config import Config
from utils.misc import setup_device, set_seed, ensure_dir
from utils.data_loader import GraphDataProcessor
from models.gat import GATEncoder
from rl.env import InfluenceEnvironment
from rl.agent import PPOAgent


def select_seeds_greedy(env, node_embeddings, agent, k_seeds):
    """Greedy seed selection using trained policy (deterministic)."""
    seed_set = []
    for _ in range(k_seeds):
        valid_actions = [node for node in env.candidate_seeds if node not in seed_set]
        if not valid_actions:
            break
        result = agent.get_action(
            node_embeddings, valid_actions, seed_set, env, deterministic=True
        )
        if result[0] is not None:
            seed_set.append(result[0])
    return seed_set


def train_im_agent(env, agent, node_embeddings, cfg, k_seeds):
    """Train PPO agent for influence maximization (Paper Algorithm 4)."""
    best_seeds = []
    best_influence = 0.0
    device = agent.device
    
    env.update_candidate_seeds(k_seeds, multiplier=cfg.candidate_multiplier)
    candidate_seeds = env.candidate_seeds
    print(f"Using candidate seed set: {len(candidate_seeds)} nodes")
    
    weight_decay = cfg.heuristic_weight / cfg.rl_epochs
    
    for episode in tqdm(range(cfg.rl_epochs)):
        trajectory = []
        seed_set = []
        
        current_weight = max(0, cfg.heuristic_weight - episode * weight_decay)
        
        for step in range(k_seeds):
            valid_actions = [node for node in candidate_seeds if node not in seed_set]
            if not valid_actions:
                break
            
            # State representation for critic
            state_repr = agent.get_state_repr(node_embeddings, seed_set)
            
            # Build composite features for all candidates (Paper Eq.13)
            composite_features = agent.build_composite_features(
                node_embeddings, valid_actions, seed_set, env
            )
            
            if random.random() < current_weight:
                # Heuristic-driven action selection (hybrid strategy)
                heuristic_scores = [env.get_heuristic_score(node, seed_set) for node in valid_actions]
                if not heuristic_scores:
                    break
                best_idx = np.argmax(heuristic_scores)
                action = valid_actions[best_idx]
                action_idx = torch.tensor(best_idx, device=device)
                
                # Still compute log_prob under current policy for PPO ratio
                with torch.no_grad():
                    scores = agent.actor(composite_features) / agent.temperature
                    probs = F.softmax(scores, dim=0)
                    dist = torch.distributions.Categorical(probs)
                    log_prob = dist.log_prob(action_idx)
            else:
                # Policy-driven action selection
                action, log_prob, action_idx = agent.get_action(
                    node_embeddings, valid_actions, seed_set, env
                )
                if action is None:
                    continue
            
            if action is None or not (0 <= action < env.n_nodes):
                continue
            
            # Compute marginal influence gain reward (Paper Eq.18)
            reward = env.get_reward(seed_set, action)
            seed_set.append(action)
            
            trajectory.append({
                'candidate_features': composite_features.detach(),
                'action_idx': action_idx,
                'old_log_prob': log_prob.item(),
                'reward': reward,
                'state_repr': state_repr.detach(),
            })
        
        # PPO update on collected trajectory
        if len(trajectory) >= 2:
            loss_info = agent.update(trajectory)

        # Periodic evaluation
        if (episode + 1) % 10 == 0:
            eval_seeds = select_seeds_greedy(env, node_embeddings, agent, k_seeds)
            influences = [env.simulate_influence(eval_seeds) for _ in range(5)]
            avg_influence = np.mean(influences)
            std_influence = np.std(influences)
            
            if avg_influence > best_influence:
                best_influence = avg_influence
                best_seeds = eval_seeds.copy()
                print(f"Episode {episode+1}: Best influence {best_influence:.2f}±{std_influence:.2f}")
    
    return best_seeds, best_influence


def run_training_pipeline(k):
    cfg = Config()
    device = setup_device()
    set_seed(42)
    ensure_dir(cfg.save_dir)
    ensure_dir(cfg.results_dir)
    
    print(f"\n{'='*30} Training with k={k} seeds {'='*30}")
    processor = GraphDataProcessor(cfg.data_path, device)
    G, features, edge_index = processor.load_and_process()
    
    gat = GATEncoder(
        in_dim=features.shape[1], hidden_dim=cfg.hidden_dim,
        out_dim=cfg.embedding_dim, device=device, heads=cfg.gat_heads
    )
    gat_path = os.path.join(cfg.save_dir, 'gat_pretrained.pth')
    
    if os.path.exists(gat_path):
        gat.load_state_dict(torch.load(gat_path, map_location=device))
        print("Loaded pretrained GAT weights.")
    else:
        print("Starting GAT pretraining...")
        gat.pretrain(features, edge_index, epochs=cfg.gat_epochs,
                     lambda_smooth=cfg.lambda_smooth)
        torch.save(gat.state_dict(), gat_path)
        
    with torch.no_grad():
        gat.eval()
        node_embeddings = gat(features, edge_index)
        
    env = InfluenceEnvironment(G, features)
    agent = PPOAgent(cfg.embedding_dim, G.number_of_nodes(), device, cfg)
    
    start_time = time.time()
    best_seeds, best_influence = train_im_agent(env, agent, node_embeddings, cfg, k)
    print(f"Training time: {time.time()-start_time:.2f}s")
    
    agent.save(os.path.join(cfg.save_dir, f'ppo_agent_k{k}.pth'))
    

if __name__ == "__main__":
    for k in range(5, 51, 5):
        try:
            run_training_pipeline(k)
        except Exception as e:
            print(f"Runtime error: {str(e)}")
            import traceback
            traceback.print_exc()
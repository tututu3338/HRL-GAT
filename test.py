import os
import torch
import numpy as np
import time
import csv
from tqdm import tqdm

from utils.config import Config
from utils.misc import setup_device, set_seed, ensure_dir
from utils.data_loader import GraphDataProcessor
from models.gat import GATEncoder
from rl.env import InfluenceEnvironment
from rl.agent import PPOAgent


def evaluate_model(k_seeds):
    cfg = Config()
    device = setup_device()
    set_seed(42)
    ensure_dir(cfg.results_dir)
    
    processor = GraphDataProcessor(cfg.data_path, device)
    G, features, edge_index = processor.load_and_process()
    
    gat = GATEncoder(
        in_dim=features.shape[1], hidden_dim=cfg.hidden_dim,
        out_dim=cfg.embedding_dim, device=device, heads=cfg.gat_heads
    )
    gat_path = os.path.join(cfg.save_dir, 'gat_pretrained.pth')
    if not os.path.exists(gat_path):
        print("Please run train.py first for pretraining!")
        return
    gat.load_state_dict(torch.load(gat_path, map_location=device))
    gat.eval()
    
    with torch.no_grad():
        node_embeddings = gat(features, edge_index)
        
    env = InfluenceEnvironment(G, features)
    agent = PPOAgent(cfg.embedding_dim, G.number_of_nodes(), device, cfg)
    
    ppo_path = os.path.join(cfg.save_dir, f'ppo_agent_k{k_seeds}.pth')
    if os.path.exists(ppo_path):
        agent.load(ppo_path)
    else:
        print(f"Warning: No pretrained RL model found for K={k_seeds}, using random policy.")
        
    candidate_seeds = env.update_candidate_seeds(k_seeds, multiplier=cfg.candidate_multiplier)
    seed_set = []
    
    print(f"Selecting {k_seeds} seeds via model inference...")
    for _ in range(k_seeds):
        valid_actions = [node for node in candidate_seeds if node not in seed_set]
        if not valid_actions:
            break
        
        # Deterministic action selection for evaluation
        action, _, _ = agent.get_action(
            node_embeddings, valid_actions, seed_set, env, deterministic=True
        )
        if action is not None:
            seed_set.append(action)
    
    print("Running precise influence evaluation...")
    precise_influences = []
    for i in tqdm(range(10), desc="Precise evaluation (MC=500)"):
        precise_influences.append(env.simulate_influence(seed_set, mc_iterations=500))
        
    final_influence = np.mean(precise_influences)
    std_influence = np.std(precise_influences)
    coverage = final_influence / env.n_nodes * 100
    
    print("\n" + "="*50)
    print(f"Evaluation complete! K={k_seeds}")
    print(f"Best seed nodes: {sorted(seed_set)}")
    print(f"Precise influence (10×500 MC): {final_influence:.2f} ± {std_influence:.2f}")
    print(f"Network coverage: {coverage:.2f}%")
    print("="*50)
    
    results_file = os.path.join(cfg.results_dir, 'test_results.csv')
    file_exists = os.path.isfile(results_file)
    with open(results_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Dataset', 'k_seeds', 'Best_Seeds', 'Final_Influence', 
                           'Std_Influence', 'Coverage_Percent', 'Timestamp'])
        writer.writerow([
            os.path.basename(cfg.data_path), k_seeds, str(sorted(seed_set)), 
            f"{final_influence:.4f}", f"{std_influence:.4f}", f"{coverage:.2f}", 
            time.strftime("%Y-%m-%d %H:%M:%S")
        ])


if __name__ == "__main__":
    for k in range(5, 51, 5):
        print(f"\n{'='*30} Testing with k={k} seeds {'='*30}")
        evaluate_model(k)
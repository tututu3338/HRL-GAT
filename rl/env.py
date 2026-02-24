import numpy as np
import networkx as nx
import torch
import time
from tqdm import tqdm


class InfluenceEnvironment:
    """WIC influence maximization environment with ECMR candidate screening.
    
    Paper Section 3.1: WIC Diffusion Model
    Paper Section 4.3: Candidate Seed Set Construction via ECMR
    """
    def __init__(self, G, node_features):
        self.G = G
        self.n_nodes = G.number_of_nodes()
        self.node_features = node_features
        self.degrees = dict(G.degree())
        self.max_degree = max(self.degrees.values()) if self.degrees else 1
        
        # WIC edge probabilities: p_uv = 1/d_v (Paper Eq.1)
        self.edge_probs = {}
        for u, v in G.edges():
            prob_uv = 1.0 / max(G.degree(v), 1)
            prob_vu = 1.0 / max(G.degree(u), 1)
            self.edge_probs[(u, v)] = prob_uv
            self.edge_probs[(v, u)] = prob_vu
        
        # Precompute clustering coefficients for all nodes
        print("Computing clustering coefficients...")
        self.clustering_coeffs = nx.clustering(G)
            
        print("Computing ECMR scores for all nodes...")
        self.ecmr_scores = {}
        for node in tqdm(list(self.G.nodes()), desc="Computing ECMR"):
            self.ecmr_scores[node] = self.calculate_ecmr(node)
        self.ecmr_sorted_nodes = sorted(
            self.ecmr_scores.items(), key=lambda x: x[1], reverse=True
        )
        self.max_ecmr = max(self.ecmr_scores.values()) if self.ecmr_scores else 1.0
        print("ECMR computation complete.")
        
        self.candidate_seeds = self.build_candidate_seed_set()
            
    def calculate_ecmr(self, node):
        """Compute ECMR score (Paper Eq.14-16).
        
        ECMR(v) = (1 + I_1(v) + I_2(v)) · (d_v/d_max + (1 - C_v))
        """
        # I_1(v): one-hop expected influence (Paper Eq.14)
        i_one = 0.0
        for neighbor in self.G.neighbors(node):
            i_one += self.edge_probs.get((node, neighbor), 0)
        
        # I_2(v): two-hop expected influence with η=0.5 (Paper Eq.15)
        i_two = 0.0
        for neighbor in self.G.neighbors(node):
            p_vu = self.edge_probs.get((node, neighbor), 0)
            for second_neighbor in self.G.neighbors(neighbor):
                if second_neighbor != node:
                    p_uw = self.edge_probs.get((neighbor, second_neighbor), 0)
                    i_two += 0.5 * p_vu * p_uw
        
        # ECMR (Paper Eq.16) — equal weights for degree and clustering factors
        degree_factor = self.degrees[node] / self.max_degree
        cluster_coef = self.clustering_coeffs[node]
        ecmr = (1 + i_one + i_two) * (degree_factor + (1 - cluster_coef))
        return ecmr
    
    def build_candidate_seed_set(self):
        k_estimate = max(50, int(self.n_nodes * 0.2)) 
        candidate_size = min(k_estimate, self.n_nodes)
        candidates = [node for node, _ in self.ecmr_sorted_nodes[:candidate_size]]
        print(f"Built candidate seed set: {len(candidates)} nodes")
        return candidates
    
    def update_candidate_seeds(self, k, multiplier):
        """Budget-adaptive candidate pool (Paper Eq.17): |C| = c·k"""
        candidate_size = min(multiplier * k, self.n_nodes)
        self.candidate_seeds = [node for node, _ in self.ecmr_sorted_nodes[:candidate_size]]
        print(f"Updated candidate seed set: {len(self.candidate_seeds)} nodes")
        return self.candidate_seeds
            
    def simulate_influence(self, seed_nodes, mc_iterations=10):
        """Monte Carlo influence estimation under WIC (Paper Algorithm 5)."""
        if not seed_nodes:
            return 0.0
        
        valid_seeds = []
        for n in seed_nodes:
            if isinstance(n, torch.Tensor):
                n = n.detach().cpu().item()
            elif isinstance(n, np.ndarray):
                n = n.item()
            n = int(n)
            if 0 <= n < self.n_nodes and n in self.G:
                valid_seeds.append(n)
        
        if not valid_seeds:
            return 0.0
        
        total_influence = 0.0
        for iter_idx in range(mc_iterations):
            random_seed = int(time.time() * 1000) % (2**32) + iter_idx
            np.random.seed(random_seed)
            
            activated = set(valid_seeds)
            frontier = set(valid_seeds)
            
            while frontier:
                new_frontier = set()
                for node in frontier:
                    neighbors = set(self.G.neighbors(node)) - activated
                    if neighbors:
                        probs = np.random.random(len(neighbors))
                        edge_probs = np.array([1.0 / max(self.G.degree(n), 1) for n in neighbors])
                        activated_nodes = np.array(list(neighbors))[probs < edge_probs]
                        new_frontier.update(activated_nodes.tolist())
                        activated.update(activated_nodes)
                frontier = new_frontier
            total_influence += len(activated)
        
        return total_influence / mc_iterations
        
    def get_reward(self, seed_set, new_seed):
        """Marginal influence gain reward (Paper Eq.18): r_t = σ(S_t ∪ {a_t}) - σ(S_t)"""
        prev_influence = self.simulate_influence(seed_set) if seed_set else 0.0
        new_influence = self.simulate_influence(seed_set + [new_seed])
        return new_influence - prev_influence
    
    def get_heuristic_score(self, node, current_seeds):
        """Heuristic score for hybrid selection strategy."""
        if node not in self.G:
            return -float('inf')
            
        degree_score = self.degrees[node] / self.max_degree
        cluster_coef = self.clustering_coeffs[node]
            
        overlap_penalty = 0
        if current_seeds:
            common_neighbors = 0
            for seed in current_seeds:
                if self.G.has_edge(node, seed):
                    common_neighbors += 1
            overlap_penalty = common_neighbors / len(current_seeds)
        
        score = 0.6 * degree_score + 0.2 * cluster_coef - 0.2 * overlap_penalty
        return score

    def get_structural_features(self, node):
        """Get static structural features ψ_i for a node (Paper Eq.13).
        
        Returns [degree_norm, clustering_coef, ecmr_norm]
        """
        degree_norm = self.degrees[node] / self.max_degree
        cluster_coef = self.clustering_coeffs.get(node, 0.0)
        ecmr_norm = self.ecmr_scores.get(node, 0.0) / max(self.max_ecmr, 1e-8)
        return [degree_norm, cluster_coef, ecmr_norm]
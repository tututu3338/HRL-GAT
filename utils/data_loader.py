import torch
import numpy as np
import pandas as pd
import networkx as nx

class GraphDataProcessor:
    def __init__(self, file_path, device):
        self.file_path = file_path
        self.device = device
        
    def load_and_process(self):
        print("正在加载数据...")
        edges = pd.read_csv(self.file_path).values
        
        unique_nodes = set()
        for u, v in edges:
            unique_nodes.add(u)
            unique_nodes.add(v)
        
        node_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted(unique_nodes))}
        
        G = nx.Graph()
        G.add_nodes_from(range(len(node_mapping)))
        mapped_edges = [(node_mapping[u], node_mapping[v]) for u, v in edges]
        G.add_edges_from(mapped_edges)
        
        print("计算节点特征...")
        degrees = dict(G.degree())
        clustering = nx.clustering(G)
        pagerank = nx.pagerank(G)
        
        features = []
        for i in range(len(node_mapping)):
            features.append([
                degrees[i],
                clustering[i],
                pagerank[i],
                np.log1p(degrees[i])
            ])
        
        features = np.array(features, dtype=np.float32)
        features = torch.from_numpy(features).to(self.device)
        
        # 归一化特征
        mean = features.mean(dim=0, keepdim=True)
        std = features.std(dim=0, keepdim=True) + 1e-8
        features = (features - mean) / std
        
        edge_index = torch.tensor(mapped_edges, dtype=torch.long, device=self.device).t()
        
        n_nodes = G.number_of_nodes()
        max_idx = edge_index.max().item()
        min_idx = edge_index.min().item()
        
        if max_idx >= n_nodes or min_idx < 0:
            print(f"警告: 边索引范围 [{min_idx}, {max_idx}] 超出节点范围 [0, {n_nodes})")
        
        print(f"图处理完成: {n_nodes}个节点，{G.number_of_edges()}条边")
        return G, features, edge_index
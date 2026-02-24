class Config:
    def __init__(self):
        # Path configuration
        self.data_path = 'data/Email.csv'
        self.save_dir = 'checkpoints'
        self.results_dir = 'results'
        
        # GAT configuration
        self.embedding_dim = 64
        self.hidden_dim = 128
        self.gat_heads = 8
        self.gat_epochs = 500
        self.lambda_smooth = 0.1       # Smoothing loss weight
        
        # RL configuration
        self.k_seeds = 50              # Maximum seed budget
        self.candidate_multiplier = 6  # Candidate pool multiplier c 
        self.rl_epochs = 50            # Number of training episodes
        self.batch_size = 64
        self.lr = 3e-4
        self.gamma = 0.99              # Discount factor
        self.gae_lambda = 0.95         # GAE lambda
        self.clip_ratio = 0.2          # PPO clip epsilo
        self.max_grad_norm = 0.5
        self.temperature = 0.5         # Action sampling temperature
        self.heuristic_weight = 0.4    # Initial heuristic weight for hybrid strategy
        
        # PPO-specific
        self.ppo_epochs = 4            # Number of PPO update epochs per trajectory
        self.entropy_coef = 0.01       # Entropy regularization coefficient η
        self.value_coef = 0.5          # Value loss coefficient β

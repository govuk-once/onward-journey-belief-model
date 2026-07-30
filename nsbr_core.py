import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuroSymbolicBeliefOptimizer(nn.Module):
    def __init__(self, embedding_dim=1536, num_classes=3, initial_vectors=None):
        super().__init__()
        # Initialize continuous destination vectors
        if initial_vectors is not None:
            self.dest_vectors = nn.Parameter(initial_vectors.clone())
        else:
            self.dest_vectors = nn.Parameter(torch.randn(num_classes, embedding_dim))
        
        # Freeze the temperature
        self.temperature = nn.Parameter(torch.tensor([15.0]), requires_grad=False)

    def forward(self, x):
        """Calculates the Belief State (Probabilities) via scaled cosine similarity."""
        x_norm = F.normalize(x, p=2, dim=1)
        v_norm = F.normalize(self.dest_vectors, p=2, dim=1)
        
        # Compute Cosine Similarity scaled by temperature
        cos_sim = torch.matmul(x_norm, v_norm.T)
        logits = cos_sim * self.temperature
        
        return logits, v_norm

class NSBRLoss(nn.Module):
    """Encapsulates the multi-objective BPTT loss formulation."""
    def __init__(self, config, beta=0.1, lambda_decay=0.8):
        super().__init__()
        self.config = config
        self.beta = beta
        self.lambda_decay = lambda_decay

    def forward(self, t1_raw_sim, t2_raw_sim, targets):
        """Computes parameterized Bayesian loss bridging Safety, Yield, Temporal, and Sim constraints."""
        prior_t1 = torch.ones_like(t1_raw_sim) / 3.0
        unnorm_t1 = torch.exp(t1_raw_sim / self.beta) * (prior_t1 ** self.lambda_decay)
        t1_probs = unnorm_t1 / torch.sum(unnorm_t1, dim=-1, keepdim=True)
        t1_ent = -torch.sum(t1_probs * torch.log2(t1_probs + 1e-9), dim=-1)
        
        prior_t2 = t1_probs 
        unnorm_t2 = torch.exp(t2_raw_sim / self.beta) * (prior_t2 ** self.lambda_decay)
        t2_probs = unnorm_t2 / torch.sum(unnorm_t2, dim=-1, keepdim=True)
        t2_ent = -torch.sum(t2_probs * torch.log2(t2_probs + 1e-9), dim=-1)
        
        id_mask = (targets != 2)
        safety_loss = F.nll_loss(torch.log(t2_probs + 1e-9), targets)
        
        if id_mask.any():
            yield_loss = F.relu(t2_ent[id_mask] - self.config["h_yield"]).mean()
            temporal_loss = F.relu(self.config["h_temp"] - t1_ent[id_mask]).mean()
            correct_sims = t2_raw_sim[id_mask, targets[id_mask]]
            
            sim_target = self.config.get("min_sim", 0.25) + 0.01 
            sim_loss = F.relu(sim_target - correct_sims).mean()
        else:
            yield_loss = temporal_loss = sim_loss = torch.tensor(0.0).to(targets.device)
            
        total_loss = (self.config["w_safe"] * safety_loss) + \
                     (self.config["w_yield"] * yield_loss) + \
                     (self.config["w_temp"] * temporal_loss) + \
                     (self.config["w_sim"] * sim_loss)
                     
        return total_loss
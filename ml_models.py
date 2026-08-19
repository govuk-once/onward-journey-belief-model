import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F

class MultinomialLogisticRegression(nn.Module):
    def __init__(self, input_dim=1024, num_classes=3):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)

class SmallMLP(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=256, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)

def calculate_gated_metrics(logits_t1, logits_t2, targets, temperature, entropy_gate):
    """Applies temperature scaling and calculates DRR, DPR, SAR, FAR."""
    
    # 1. Temperature-Scaled Softmax
    probs_t1 = F.softmax(logits_t1 / temperature, dim=-1)
    probs_t2 = F.softmax(logits_t2 / temperature, dim=-1)
    
    # 2. Shannon Entropy
    ent_t1 = -torch.sum(probs_t1 * torch.log2(probs_t1 + 1e-9), dim=-1)
    ent_t2 = -torch.sum(probs_t2 * torch.log2(probs_t2 + 1e-9), dim=-1)
    
    # 3. Class Predictions
    preds_t1 = torch.argmax(probs_t1, dim=-1)
    preds_t2 = torch.argmax(probs_t2, dim=-1)
    
    # 4. Routing Gate Logic
    t1_acts = (ent_t1 <= entropy_gate) & (preds_t1 != 2)
    t2_acts = (~t1_acts) & (ent_t2 <= entropy_gate) & (preds_t2 != 2)
    
    # 5. Confusion Matrix (Matching NSBR methodology)
    is_in_domain = (targets != 2)
    is_ood = (targets == 2)
    
    tp = ((t1_acts & (preds_t1 == targets) & is_in_domain).sum() + 
          (t2_acts & (preds_t2 == targets) & is_in_domain).sum()).item()
          
    fp_wrong_dest = ((t1_acts & (preds_t1 != targets) & is_in_domain).sum() + 
                     (t2_acts & (preds_t2 != targets) & is_in_domain).sum()).item()
    fp_ood_leak = (t1_acts & is_ood).sum().item() + (t2_acts & is_ood).sum().item()
    fp = fp_wrong_dest + fp_ood_leak
    
    no_action = (~t1_acts) & (~t2_acts)
    tn = (no_action & is_ood).sum().item()
    fn = (no_action & is_in_domain).sum().item()
    
    # 6. Final Rates
    total = len(targets)
    in_domain_total = is_in_domain.sum().item()
    ood_total = is_ood.sum().item()
    
    dpr = ((tp + tn + fn) / total) * 100 if total > 0 else 0.0
    drr = (tp / in_domain_total) * 100 if in_domain_total > 0 else 0.0
    sar = (tn / ood_total) * 100 if ood_total > 0 else 0.0
    far = (fn / in_domain_total) * 100 if in_domain_total > 0 else 0.0
    
    return {"DPR": dpr, "DRR": drr, "SAR": sar, "FAR": far}
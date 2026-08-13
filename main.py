import os
import torch
from visualizer import NSBRVisualizer
from experiment_pipeline import ExperimentPipeline

# 1. PHASE-SPECIFIC HYPERPARAMETER CONFIGURATIONS (CALIBRATED INFERENCE GATES)

PHASE_I_VARIANTS = {
    # REGIME 1: ULTRA-SAFETY & ZERO-LEAKAGE (High Margin, Controlled Gates)
    "BPTT_Ultra_Safety": {
        "w_safe": 8.0, "w_yield": 1.0, "w_temp": 2.0, "w_sim": 4.0, 
        "h_yield": 0.70, "h_temp": 0.60, "min_sim": 0.25, 
        "eval_gate": 0.90, "eval_sim": 0.20, "beta": 0.05
    },
    "BPTT_Low_Entropy_Focus": {
        "w_safe": 4.0, "w_yield": 6.0, "w_temp": 1.0, "w_sim": 2.0, 
        "h_yield": 0.40, "h_temp": 0.50, "min_sim": 0.20, 
        "eval_gate": 1.10, "eval_sim": 0.15, "beta": 0.05
    },

    # REGIME 2: OPTIMAL PARETO & BALANCED (The "Knee" Region)
    "BPTT_Sim_Strict": {  # Primary Recommended Configuration (~90% DRR @ ~97% QPR)
        "w_safe": 3.0, "w_yield": 3.0, "w_temp": 1.0, "w_sim": 6.0, 
        "h_yield": 0.60, "h_temp": 0.70, "min_sim": 0.30, 
        "eval_gate": 1.40, "eval_sim": 0.15, "beta": 0.05
    },
    "BPTT_Balanced": {
        "w_safe": 2.0, "w_yield": 3.0, "w_temp": 1.0, "w_sim": 2.0, 
        "h_yield": 0.60, "h_temp": 0.70, "min_sim": 0.15, 
        "eval_gate": 1.10, "eval_sim": 0.15, "beta": 0.08
    },
    "BPTT_Dual_Max": {
        "w_safe": 5.0, "w_yield": 5.0, "w_temp": 2.0, "w_sim": 3.0, 
        "h_yield": 0.55, "h_temp": 0.60, "min_sim": 0.20, 
        "eval_gate": 1.00, "eval_sim": 0.15, "beta": 0.08
    },
    "BPTT_Equal_Weights": {
        "w_safe": 1.0, "w_yield": 1.0, "w_temp": 1.0, "w_sim": 1.0, 
        "h_yield": 0.60, "h_temp": 0.70, "min_sim": 0.15, 
        "eval_gate": 1.10, "eval_sim": 0.15, "beta": 0.10
    },

    # REGIME 3: FAST LOCK & RELAXED MARGINS (Early Turn / Speed Focus)
    "BPTT_Fast_Turn1": {
        "w_safe": 2.0, "w_yield": 3.0, "w_temp": 6.0, "w_sim": 2.0, 
        "h_yield": 0.60, "h_temp": 0.40, "min_sim": 0.15, 
        "eval_gate": 1.20, "eval_sim": 0.10, "beta": 0.08
    },
    "BPTT_Fast_Lock_Relaxed_Sim": {
        "w_safe": 1.5, "w_yield": 5.0, "w_temp": 4.0, "w_sim": 0.2, 
        "h_yield": 0.50, "h_temp": 0.50, "min_sim": 0.05, 
        "eval_gate": 1.25, "eval_sim": 0.05, "beta": 0.10
    },
    "BPTT_Sim_Relaxed": {
        "w_safe": 2.0, "w_yield": 4.0, "w_temp": 1.0, "w_sim": 0.5, 
        "h_yield": 0.60, "h_temp": 0.70, "min_sim": 0.10, 
        "eval_gate": 1.15, "eval_sim": 0.10, "beta": 0.10
    },
    "BPTT_Temp_Relaxed": {
        "w_safe": 2.0, "w_yield": 4.0, "w_temp": 0.1, "w_sim": 2.0, 
        "h_yield": 0.60, "h_temp": 0.80, "min_sim": 0.15, 
        "eval_gate": 1.15, "eval_sim": 0.15, "beta": 0.10
    },

    # REGIME 4: HIGH-YIELD BUMP & AGGRESSIVE ROUTING
    "BPTT_Yield_Bump": {
        "w_safe": 2.0, "w_yield": 5.0, "w_temp": 1.0, "w_sim": 2.0, 
        "h_yield": 0.55, "h_temp": 0.70, "min_sim": 0.15, 
        "eval_gate": 1.20, "eval_sim": 0.10, "beta": 0.08
    },
    "BPTT_Aggressive_Yield": {
        "w_safe": 1.5, "w_yield": 8.0, "w_temp": 0.5, "w_sim": 0.5, 
        "h_yield": 0.50, "h_temp": 0.70, "min_sim": 0.10, 
        "eval_gate": 1.30, "eval_sim": 0.05, "beta": 0.10
    },

    # REGIME 5: HYPER-YIELD & UNCONSTRAINED (The Safety Cliff)
    "BPTT_Hyper_Yield_10x": {
        "w_safe": 1.0, "w_yield": 10.0, "w_temp": 0.1, "w_sim": 0.1, 
        "h_yield": 0.30, "h_temp": 0.90, "min_sim": 0.00, 
        "eval_gate": 1.10, "eval_sim": 0.00, "beta": 0.05
    },
    "BPTT_Hyper_Yield_20x": {
        "w_safe": 0.5, "w_yield": 20.0, "w_temp": 0.05, "w_sim": 0.0, 
        "h_yield": 0.20, "h_temp": 0.95, "min_sim": 0.00, 
        "eval_gate": 1.20, "eval_sim": 0.00, "beta": 0.05
    },
    "BPTT_Unconstrained_Yield": {
        "w_safe": 0.1, "w_yield": 15.0, "w_temp": 0.0, "w_sim": 0.0, 
        "h_yield": 0.10, "h_temp": 1.00, "min_sim": -0.10, 
        "eval_gate": 1.40, "eval_sim": -0.05, "beta": 0.10
    }
}

PHASE_II_VARIANTS = {
    # Phase II inherits the full calibrated spectrum for direct comparative evaluation
    **PHASE_I_VARIANTS
}

def main():
    # Automatically scaffold project output directories
    for directory in ["data", "cache", "results"]:
        os.makedirs(directory, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Initializing NSBR Pipeline. Compute Engine: {device}")

    p1_path = os.path.join('data', 'synthetic_corpus_phase1.json')
    p2_path = os.path.join('data', 'synthetic_corpus_phase2.json')
    
    if not os.path.exists(p1_path) or not os.path.exists(p2_path):
        print("\n[!] Error: Required datasets not found in the 'data/' directory.")
        print("Please run your data generators (data_gen.py and data_generator.py) before executing main.py.")
        return

    # --- PHASE I: CLEAN BASELINE ---
    print("\n" + "="*90)
    print("STARTING PHASE I: CLEAN BASELINE EVALUATION")
    print("="*90)
    
    phase1_pipeline = ExperimentPipeline(
        json_path=p1_path,  
        device=device,
        phase_prefix='phase1_grouped',
        variants_config=PHASE_I_VARIANTS
    )
    
    p1_train, p1_test, p1_full, p1_tr_path, p1_te_path = phase1_pipeline.execute_monte_carlo(
        seeds=[42, 123, 777, 2026, 9999]
    )    
    NSBRVisualizer.plot_triple_results(p1_train, p1_test, p1_full, "Phase I (Clean Baseline)")

    # --- PHASE II: ADVERSARIAL CORPUS ---
    print("\n" + "="*90)
    print("STARTING PHASE II: ADVERSARIAL CORPUS DEGRADATION & RECOVERY")
    print("="*90)
    
    phase2_pipeline = ExperimentPipeline(
        json_path=p2_path,
        device=device,
        phase_prefix='phase2_grouped',
        variants_config=PHASE_II_VARIANTS
    )
    
    p2_train, p2_test, p2_full, p2_tr_path, p2_te_path = phase2_pipeline.execute_monte_carlo(
        seeds=[42, 123, 777, 2026, 9999]
    )
    NSBRVisualizer.plot_triple_results(p2_train, p2_test, p2_full, "Phase II (Adversarial Corpus)")

if __name__ == "__main__":
    main()
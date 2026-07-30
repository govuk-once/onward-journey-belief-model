import os
import torch
from visualizer import NSBRVisualizer
from experiment_pipeline import ExperimentPipeline

# 1. PHASE-SPECIFIC HYPERPARAMETER CONFIGURATIONS

PHASE_I_VARIANTS = {
    "BPTT_Balanced": {
        "w_safe": 2.0, "w_yield": 3.0, "w_temp": 1.0, "w_sim": 2.0, 
        "h_yield": 0.60, "h_temp": 0.70, "min_sim": 0.15, "eval_gate": 0.85, "beta": 0.10
    },
    "BPTT_Yield_Bump": {
        "w_safe": 2.0, "w_yield": 5.0, "w_temp": 1.0, "w_sim": 2.0, 
        "h_yield": 0.55, "h_temp": 0.70, "min_sim": 0.15, "eval_gate": 0.85, "beta": 0.10
    },
    "BPTT_Sim_Relaxed": {
        "w_safe": 2.0, "w_yield": 4.0, "w_temp": 1.0, "w_sim": 0.5, 
        "h_yield": 0.60, "h_temp": 0.70, "min_sim": 0.10, "eval_gate": 0.85, "beta": 0.10
    },
    "BPTT_Temp_Relaxed": {
        "w_safe": 2.0, "w_yield": 4.0, "w_temp": 0.1, "w_sim": 2.0, 
        "h_yield": 0.60, "h_temp": 0.80, "min_sim": 0.15, "eval_gate": 0.85, "beta": 0.10
    },
    "BPTT_Aggressive_Yield": {
        "w_safe": 1.5, "w_yield": 8.0, "w_temp": 0.5, "w_sim": 0.5, 
        "h_yield": 0.50, "h_temp": 0.70, "min_sim": 0.10, "eval_gate": 0.85, "beta": 0.10
    }
}

PHASE_II_VARIANTS = {
    "BPTT_Balanced":         {"w_safe": 2.0, "w_yield": 3.0, "w_temp": 1.0, "w_sim": 2.0, "h_yield": 0.60, "h_temp": 0.70, "min_sim": 0.25, "eval_gate": 0.66, "beta": 0.10},
    "BPTT_Yield_Bump":       {"w_safe": 2.0, "w_yield": 4.0, "w_temp": 1.0, "w_sim": 2.0, "h_yield": 0.55, "h_temp": 0.70, "min_sim": 0.25, "eval_gate": 0.66, "beta": 0.10},
    "BPTT_Sim_Relaxed":      {"w_safe": 2.0, "w_yield": 3.0, "w_temp": 1.0, "w_sim": 1.5, "h_yield": 0.60, "h_temp": 0.70, "min_sim": 0.10, "eval_gate": 0.75, "beta": 0.15}, 
    "BPTT_Temp_Relaxed":     {"w_safe": 2.0, "w_yield": 3.0, "w_temp": 0.5, "w_sim": 2.0, "h_yield": 0.60, "h_temp": 0.80, "min_sim": 0.25, "eval_gate": 0.81, "beta": 0.15},
    "BPTT_Aggressive_Yield": {"w_safe": 1.5, "w_yield": 8.0, "w_temp": 0.5, "w_sim": 1.5, "h_yield": 0.50, "h_temp": 0.70, "min_sim": 0.10, "eval_gate": 0.81, "beta": 0.15}
}

def main():
    # Automatically scaffold the clean project structure
    for directory in ["data", "cache", "results"]:
        os.makedirs(directory, exist_ok=True)

    # Detect optimal compute hardware
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Initializing NSBR Pipeline. Compute Engine: {device}")

    # --- Dataset Validation Check ---
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
        phase_prefix='phase1_grouped', # Updated to force fresh caching for the holdout split
        variants_config=PHASE_I_VARIANTS
    )
    
    # Execute Monte Carlo across 5 academic seeds
# Unpack all 5 returned items (including trace paths)
    p1_train, p1_test, p1_full, p1_tr_path, p1_te_path = phase1_pipeline.execute_monte_carlo(
    seeds=[42, 123, 777, 2026, 9999]
)    
    # Visualize the aggregated results
    NSBRVisualizer.plot_triple_results(p1_train, p1_test, p1_full, "Phase I (Clean Baseline)")


    # --- PHASE II: ADVERSARIAL CORPUS ---
    print("\n" + "="*90)
    print("STARTING PHASE II: ADVERSARIAL CORPUS DEGRADATION & RECOVERY")
    print("="*90)
    
    phase2_pipeline = ExperimentPipeline(
        json_path=p2_path,
        device=device,
        phase_prefix='phase2_grouped', # Updated to force fresh caching for the holdout split
        variants_config=PHASE_II_VARIANTS
    )
    
    # Execute Monte Carlo across 5 academic seeds
    p2_train, p2_test, p2_full, p2_tr_path, p2_te_path = phase2_pipeline.execute_monte_carlo(seeds=[42, 123, 777, 2026, 9999])
    
    # Visualize the aggregated results
    NSBRVisualizer.plot_triple_results(p2_train, p2_test, p2_full, "Phase II (Adversarial Corpus)")

if __name__ == "__main__":
    main()
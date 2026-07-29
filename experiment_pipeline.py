import os
import sys
import json
import time
import torch
import numpy as np
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor, as_completed

from nsbr_core import NeuroSymbolicBeliefOptimizer, NSBRLoss
from belief import embed, KNOWLEDGE_BASE_TIERS, LLMBaselineAgent

class ExperimentPipeline:
    """
    Executes the multi-stage evaluation pipeline for Neuro-Symbolic Belief Routing.
    Handles data stratification, LLM baseline testing, manual geometric checks, 
    and BPTT learned parameter optimization.
    """
    
    OPTIMIZED_GATES = {
        "Low":                 {"min_sim": 0.25, "max_entropy": 0.80},
        "Medium":              {"min_sim": 0.25, "max_entropy": 0.66}, 
        "High":                {"min_sim": 0.25, "max_entropy": 0.86},
        "Extreme_Separation":  {"min_sim": 0.25, "max_entropy": 0.39},
        "Asymmetric_Taxonomy": {"min_sim": 0.25, "max_entropy": 1.13},
        "Hierarchical_Tokens": {"min_sim": 0.25, "max_entropy": 0.66},
        "Context_Insulated":   {"min_sim": 0.25, "max_entropy": 0.60},
        "Negative_Sampling":   {"min_sim": 0.25, "max_entropy": 1.07}
    }

    def __init__(self, json_path, device, phase_prefix, variants_config):
        self.json_path = json_path
        self.device = device
        self.phase_prefix = phase_prefix
        self.variants_config = variants_config

    def prepare_data(self, seed):
        """Loads and 4-way stratifies corpus using a Grouped or Stratified Holdout Split, handling Titan caching."""
        print(f"\nLoading corpus from {self.json_path} (Seed: {seed})...")
        with open(self.json_path, 'r', encoding='utf-8') as f:
            corpus = json.load(f)
            
        texts_t1, texts_t2, raw_data = [], [], []
        labels_3way, stratify_labels_4way = [], []
        
        for item in corpus:
            traj_id = item.get("id", "")
            gt = item.get("ground_truth", "")
            dialogue = item.get("dialogue", [])
            
            texts_t1.append(dialogue[0])
            texts_t2.append(dialogue[1] if len(dialogue) > 1 else dialogue[0])
            
            tier = "OOD_Trap"
            if gt.upper() != "OOD" and "ood" not in gt.lower():
                tier = "Ambiguous_ID" if len(dialogue) > 1 else "Direct_ID"
            item["operational_tier"] = tier
            raw_data.append(item)
            
            if "dvla" in gt.lower():
                labels_3way.append(0)
                stratify_labels_4way.append(0)
            elif "dvsa" in gt.lower():
                labels_3way.append(1)
                stratify_labels_4way.append(1)
            else:
                labels_3way.append(2)
                stratify_labels_4way.append(2 if "near" in traj_id.lower() else 3)
                    
        # --- STRATIFIED GROUP SPLIT LOGIC ---
        train_idx, test_idx = [], []
        has_groups = any("base_intent_id" in item for item in raw_data)
        
        if has_groups:
            print("Grouping Key Detected: Performing Stratified Intent Group Split...")
            class_to_indices = {0: [], 1: [], 2: [], 3: []}
            for i, label in enumerate(stratify_labels_4way):
                class_to_indices[label].append(i)
                
            for cls, idxs in class_to_indices.items():
                cls_groups = np.array([raw_data[i].get("base_intent_id", raw_data[i]["id"]) for i in idxs])
                unique_groups = np.unique(cls_groups)
                
                np.random.seed(seed)
                np.random.shuffle(unique_groups)
                split_point = int(len(unique_groups) * 0.8)
                train_groups = set(unique_groups[:split_point])
                
                for i in idxs:
                    if raw_data[i].get("base_intent_id", raw_data[i]["id"]) in train_groups:
                        train_idx.append(i)
                    else:
                        test_idx.append(i)
        else:
            print("No 'base_intent_id' found: Performing Standard 4-Way Stratified Split...")
            from sklearn.model_selection import train_test_split
            indices = np.arange(len(raw_data))
            train_idx, test_idx = train_test_split(
                indices, test_size=0.2, stratify=stratify_labels_4way, random_state=seed
            )
            
        train_idx, test_idx = np.array(train_idx), np.array(test_idx)
        print(f"Data Split Complete: {len(train_idx)} Train, {len(test_idx)} Test")

        cache_file = os.path.join("cache", f"{self.phase_prefix}_titan_10k_embeddings_cache.pt")
        os.makedirs("cache", exist_ok=True)
        os.makedirs("results", exist_ok=True)

        if os.path.exists(cache_file):
            print(f"FAST LOAD: Found cached Titan V2 embeddings at '{cache_file}'.")
            cache = torch.load(cache_file, weights_only=True)
            all_t1_embs = cache['t1'].to(self.device)
            all_t2_embs = cache['t2'].to(self.device)
        else:
            print("No cache found. Processing via ThreadPoolExecutor...")
            def embed_concurrently(texts, desc):
                results = [None] * len(texts)
                with ThreadPoolExecutor(max_workers=10) as executor:
                    future_to_idx = {executor.submit(embed, text): i for i, text in enumerate(texts)}
                    for future in tqdm(as_completed(future_to_idx), total=len(texts), desc=desc):
                        results[future_to_idx[future]] = future.result()
                return results

            t1_vectors = embed_concurrently(texts_t1, "Embedding Turn 1")
            t2_vectors = embed_concurrently(texts_t2, "Embedding Turn 2")
            
            all_t1_embs = torch.tensor(np.array(t1_vectors), dtype=torch.float32)
            all_t2_embs = torch.tensor(np.array(t2_vectors), dtype=torch.float32)
            torch.save({'t1': all_t1_embs, 't2': all_t2_embs}, cache_file)
            all_t1_embs, all_t2_embs = all_t1_embs.to(self.device), all_t2_embs.to(self.device)
        
        all_labels = torch.tensor(labels_3way, dtype=torch.long).to(self.device)
        
        train_t1, train_t2, train_y = all_t1_embs[train_idx], all_t2_embs[train_idx], all_labels[train_idx]
        train_raw = [raw_data[i] for i in train_idx]
        test_t1, test_t2, test_y = all_t1_embs[test_idx], all_t2_embs[test_idx], all_labels[test_idx]
        test_raw = [raw_data[i] for i in test_idx]
        
        return train_t1, train_t2, train_y, train_raw, test_t1, test_t2, test_y, test_raw, all_t1_embs.shape[1]

    def _evaluate_single_llm_trajectory(self, item, max_retries=5):
        gt_intent = item["ground_truth"]
        dialogue_turns = item["dialogue"]
        
        # Switched back to Haiku for massive speed improvements
        agent = LLMBaselineAgent(model_name="anthropic.claude-3-7-sonnet-20250219-v1:0", aws_region="eu-west-2")
        
        outcome = "SA"
        conversation_context = ""
        last_res = ""
        turns_taken = 0
        
        # NEW: Initialize the history array to capture turn-by-turn interactions
        trajectory_history = []
        
        # Unwind the synthetic trajectory chronologically
        for turn in dialogue_turns:
            turns_taken += 1
            conversation_context += f"{turn}\n"
            attempts = 0
            
            while True:
                attempts += 1
                try:
                    # Send the accumulated history up to this point
                    last_res = agent.send_message(conversation_context)
                    break  # Success!
                except Exception as exc:
                    exc_str = str(exc)
                    # Detect expired credentials or connection drops
                    if any(err_kw in exc_str for err_kw in [
                        "ExpiredToken", "Credentials", "expired", "UnrecognizedClientException", 
                        "AccessDenied", "EndpointConnectionError"
                    ]):
                        raise RuntimeError(f"FATAL_AWS_CREDENTIAL_ERROR: {exc_str}")

                    # Exponential backoff on rate throttling
                    if attempts <= max_retries and any(t_kw in exc_str for t_kw in ["ThrottlingException", "Too many requests", "RateExceeded"]):
                        sleep_time = (2 ** attempts) * 0.5
                        time.sleep(sleep_time)
                        continue
                    else:
                        # Append the error turn cleanly before returning
                        trajectory_history.append({
                            "turn": turns_taken,
                            "user_prompt": turn,
                            "llm_response": f"ERROR: {exc_str}"
                        })
                        
                        return item["id"], {
                            "history": trajectory_history,
                            "outcome": "MR",
                            "turns_taken": turns_taken
                        }

            # Success block: Capture this specific turn's interaction in the structured array
            trajectory_history.append({
                "turn": turns_taken,
                "user_prompt": turn,
                "llm_response": last_res
            })

            res_clean = last_res.lower()
            
            # Did the LLM decide to execute a route?
            if "decision executed" in res_clean or "connect_to" in res_clean:
                # Case-insensitive ground truth matching
                gt_clean = gt_intent.lower().replace("connect_to_", "").strip()
                if gt_intent.upper() == "OOD" or gt_clean == "ood":
                    outcome = "MR"
                elif gt_clean in res_clean:
                    outcome = "SR"
                else:
                    outcome = "MR"
                break # Stop evaluating this trajectory; the gate closed.
            else:
                # The LLM abstained and asked a clarifying question.
                # Append its question to the context and loop to the next user turn.
                conversation_context += f"Assistant: {last_res}\n"
                
        # If the loop finishes and it never routed, check if it was supposed to be OOD
        if outcome == "SA" and (gt_intent.upper() == "OOD" or gt_intent == "ood"):
            outcome = "TN"
            
        # Final return outputs the structured history array instead of the raw strings
        return item["id"], {
            "history": trajectory_history,
            "outcome": outcome,
            "turns_taken": turns_taken
        }

    def run_llm_baseline(self, train_raw, test_raw):
        """Executes zero-shot LLM evaluations concurrently, gracefully handling credential expiration."""
        print(f"\n{'='*90}\nRUNNING ZERO-SHOT LLM BASELINE ({self.phase_prefix})\n{'='*90}")
        full_raw = train_raw + test_raw
        cache_file = os.path.join("cache", f"{self.phase_prefix}_llm_baseline_cache.json")
        cache_dict = {}
        
        if os.path.exists(cache_file):
            with open(cache_file, "r", encoding="utf-8") as f:
                cache_dict = json.load(f)
                
            # Purge poisoned cache entries from prior failed authentication runs
            purged_keys = [k for k, v in cache_dict.items() if "ERROR:" in str(v.get("llm_response", ""))]
            if purged_keys:
                print(f"Purging {len(purged_keys)} errored items from previous cache...")
                for k in purged_keys:
                    del cache_dict[k]

        missing = [item for item in full_raw if item["id"] not in cache_dict]
        
        if missing:
            print(f"Found {len(missing)} uncached items. Executing LLM calls...")
            fatal_error = None
            completed_count = 0  # Counter for checkpointing
            
            with ThreadPoolExecutor(max_workers=25) as executor:  # Using 25 workers as previously discussed
                futures = {executor.submit(self._evaluate_single_llm_trajectory, item): item for item in missing}
                
                for future in tqdm(as_completed(futures), total=len(missing), desc="LLM Inference"):
                    try:
                        traj_id, res = future.result()
                        cache_dict[traj_id] = res
                    except Exception as e:
                        if "FATAL_AWS_CREDENTIAL_ERROR" in str(e):
                            fatal_error = str(e)
                            executor.shutdown(wait=False, cancel_futures=True)
                            break
                        else:
                            print(f"\nError processing trajectory: {e}")
                    finally:
                        completed_count += 1
                        # Checkpoint every 100 iterations
                        if completed_count % 100 == 0:
                            with open(cache_file, "w", encoding="utf-8") as f:
                                json.dump(cache_dict, f, indent=4)

            # Final save state before exiting (catches the remainder)
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_dict, f, indent=4)

            if fatal_error:
                print(f"\n{'!'*90}")
                print("AWS CREDENTIAL EXPIRED OR FATAL CONNECTION ERROR ENCOUNTERED!")
                print(f"Details: {fatal_error}")
                print(f"Saved {len(cache_dict)} processed items to '{cache_file}'.")
                print("Please refresh your AWS credentials and re-run the script to resume.")
                print(f"{'!'*90}\n")
                sys.exit(1)
                
        def get_llm_traces(dataset):
            traces = []
            for item in dataset:
                if item["id"] in cache_dict:
                    cached_data = cache_dict[item["id"]]
                    dialogue = item.get("dialogue", [])
                    traces.append({
                        "trajectory_id": item["id"],
                        "base_intent_id": item.get("base_intent_id", item["id"]),
                        "operational_tier": item["operational_tier"],
                        "ground_truth": item["ground_truth"],
                        "method": "Zero-Shot LLM Baseline",
                        "user_sample": {
                            "turn_1": dialogue[0] if len(dialogue) > 0 else "",
                            "turn_2": dialogue[1] if len(dialogue) > 1 else (dialogue[0] if len(dialogue) > 0 else "")
                        },
                        "llm_interaction": cached_data.get("history", []),
                        "turns_taken": cached_data.get("turns_taken", 0),
                        "outcome": cached_data["outcome"]
                    })
            return traces

        train_traces = get_llm_traces(train_raw)
        test_traces = get_llm_traces(test_raw)
                
        def calc_metrics(dataset):
            if not dataset: return 0.0, 0.0
            total = len(dataset)
            in_domain = sum(1 for item in dataset if item["ground_truth"].upper() != "OOD")
            outcomes = [cache_dict[item["id"]]["outcome"] for item in dataset if item["id"] in cache_dict]
            sr, tn, sa = outcomes.count("SR"), outcomes.count("TN"), outcomes.count("SA")
            
            dpr = ((sr + tn + sa) / total) * 100 if total > 0 else 0.0
            drr = (sr / in_domain) * 100 if in_domain > 0 else 0.0
            return dpr, drr

        tr_qpr, tr_drr = calc_metrics(train_raw)
        te_qpr, te_drr = calc_metrics(test_raw)
        f_qpr, f_drr = calc_metrics(full_raw)
        
        print(f"{'TIER NAME':<20} | {'TRAIN DPR':<9} | {'TRAIN DRR':<9} || {'TEST DPR':<9} | {'TEST DRR':<9}")
        print(f"{'LLM Baseline':<20} | {tr_qpr:>8.1f}% | {tr_drr:>8.1f}% || {te_qpr:>8.1f}% | {te_drr:>8.1f}%")
        
        return {"Zero-Shot:\nLLM Baseline": {"QPR": tr_qpr, "DRR": tr_drr}}, \
               {"Zero-Shot:\nLLM Baseline": {"QPR": te_qpr, "DRR": te_drr}}, \
               {"Zero-Shot:\nLLM Baseline": {"QPR": f_qpr, "DRR": f_drr}}, \
               train_traces, test_traces

    @staticmethod
    def _evaluate_universal_bayesian(t1_raw_sim, t2_raw_sim, targets, gate_threshold, min_sim_threshold, beta=0.1, lambda_decay=0.8, raw_items=None, method_name=""):
        """Computes deterministic safety and yield metrics over trajectory tensors and returns deep traces."""
        prior_t1 = torch.ones_like(t1_raw_sim) / 3.0
        unnorm_t1 = torch.exp(t1_raw_sim / beta) * (prior_t1 ** lambda_decay)
        t1_probs = unnorm_t1 / torch.sum(unnorm_t1, dim=-1, keepdim=True)
        t1_entropy = -torch.sum(t1_probs * torch.log2(t1_probs + 1e-9), dim=-1)
        t1_preds = torch.argmax(t1_probs, dim=-1)
        
        prior_t2 = t1_probs 
        unnorm_t2 = torch.exp(t2_raw_sim / beta) * (prior_t2 ** lambda_decay)
        t2_probs = unnorm_t2 / torch.sum(unnorm_t2, dim=-1, keepdim=True)
        t2_entropy = -torch.sum(t2_probs * torch.log2(t2_probs + 1e-9), dim=-1)
        t2_preds = torch.argmax(t2_probs, dim=-1)
        
        t1_max_sim, _ = torch.max(t1_raw_sim, dim=-1)
        t2_max_sim, _ = torch.max(t2_raw_sim, dim=-1)
        
        t1_approved = (t1_entropy <= gate_threshold) & (t1_max_sim >= min_sim_threshold)
        t2_approved = (t2_entropy <= gate_threshold) & (t2_max_sim >= min_sim_threshold)
        
        t1_acts = t1_approved & (t1_preds != 2)
        t2_acts = (~t1_acts) & t2_approved & (t2_preds != 2)
        
        res = (t1_acts & (t1_preds == targets) & (targets != 2)).sum() + \
              (t2_acts & (t2_preds == targets) & (targets != 2)).sum()
              
        no_action = (~t1_acts) & (~t2_acts)
        tn = (no_action & (targets == 2)).sum()
        abstentions = (no_action & (targets != 2)).sum()
        
        total = len(targets)
        in_domain_total = (targets != 2).sum().item()
        
        qpr = ((res.item() + tn.item() + abstentions.item()) / total) * 100
        drr = (res.item() / in_domain_total) * 100 if in_domain_total > 0 else 0.0

        traces = []
        if raw_items is not None:
            for i in range(total):
                gt = targets[i].item()
                if t1_acts[i].item():
                    pred, turn = t1_preds[i].item(), 1
                elif t2_acts[i].item():
                    pred, turn = t2_preds[i].item(), 2
                else:
                    pred, turn = -1, 0

                if pred == gt and gt != 2:
                    outcome = "SR"
                elif pred == -1 and gt == 2:
                    outcome = "TN"
                elif pred == -1 and gt != 2:
                    outcome = "SA"
                else:
                    outcome = "MR"

                raw_item = raw_items[i]
                dialogue = raw_item.get("dialogue", [])

                traces.append({
                    "trajectory_id": raw_item["id"],
                    "base_intent_id": raw_item.get("base_intent_id", raw_item["id"]),
                    "operational_tier": raw_item["operational_tier"],
                    "ground_truth": raw_item["ground_truth"],
                    "method": method_name,
                    "user_sample": {
                        "turn_1": dialogue[0] if len(dialogue) > 0 else "",
                        "turn_2": dialogue[1] if len(dialogue) > 1 else (dialogue[0] if len(dialogue) > 0 else "")
                    },
                    "t1_probs": [round(p, 4) for p in t1_probs[i].tolist()],
                    "t1_entropy": round(t1_entropy[i].item(), 4),
                    "t1_max_sim": round(t1_max_sim[i].item(), 4),
                    "t2_probs": [round(p, 4) for p in t2_probs[i].tolist()],
                    "t2_entropy": round(t2_entropy[i].item(), 4),
                    "t2_max_sim": round(t2_max_sim[i].item(), 4),
                    "routed_turn": turn,
                    "prediction_class": pred,
                    "outcome": outcome
                })

        return qpr, drr, traces

    def run_manual_evaluation(self, train_t1, train_t2, train_y, train_raw, test_t1, test_t2, test_y, test_raw, seed):
        """Runs geometric checks and builds train and test traces."""
        print(f"\n{'='*90}\nRUNNING MANUAL EVALUATIONS (TITAN V2 + BAYESIAN MATH)\n{'='*90}")
        train_res, test_res, full_res = {}, {}, {}
        all_tr_traces, all_te_traces = [], []
        t1_norm_test, t2_norm_test = F.normalize(test_t1, p=2, dim=1), F.normalize(test_t2, p=2, dim=1)
        t1_norm_train, t2_norm_train = F.normalize(train_t1, p=2, dim=1), F.normalize(train_t2, p=2, dim=1)
        
        print(f"{'TIER NAME':<20} | {'TRAIN QPR':<9} | {'TRAIN DRR':<9} || {'TEST QPR':<9} | {'TEST DRR':<9}")
        print("-" * 75)

        for tier_name, gate_rules in self.OPTIMIZED_GATES.items():
            if tier_name not in KNOWLEDGE_BASE_TIERS: continue
            kb = KNOWLEDGE_BASE_TIERS[tier_name]
            dest_embs = torch.tensor(np.array([embed(kb["DVLA"]), embed(kb["DVSA"]), embed(kb["OOD"])]), dtype=torch.float32).to(self.device)
            dest_norm = F.normalize(dest_embs, p=2, dim=1)
            
            t1_sim_train, t2_sim_train = torch.matmul(t1_norm_train, dest_norm.T), torch.matmul(t2_norm_train, dest_norm.T)
            t1_sim_test, t2_sim_test = torch.matmul(t1_norm_test, dest_norm.T), torch.matmul(t2_norm_test, dest_norm.T)
            
            method_name = f"Handcrafted_{tier_name}"
            
            tr_qpr, tr_drr, tr_traces = self._evaluate_universal_bayesian(
                t1_sim_train, t2_sim_train, train_y, gate_rules["max_entropy"], gate_rules["min_sim"], 
                raw_items=train_raw, method_name=method_name
            )
            te_qpr, te_drr, te_traces = self._evaluate_universal_bayesian(
                t1_sim_test, t2_sim_test, test_y, gate_rules["max_entropy"], gate_rules["min_sim"], 
                raw_items=test_raw, method_name=method_name
            )

            all_tr_traces.extend(tr_traces)
            all_te_traces.extend(te_traces)
            
            print(f"{tier_name:<20} | {tr_qpr:>8.1f}% | {tr_drr:>8.1f}% || {te_qpr:>8.1f}% | {te_drr:>8.1f}%")

            display = f"Handcrafted:\n{tier_name}"
            train_res[display] = {"QPR": tr_qpr, "DRR": tr_drr}
            test_res[display] = {"QPR": te_qpr, "DRR": te_drr}
            full_res[display] = {"QPR": te_qpr, "DRR": te_drr} 
            
        return train_res, test_res, full_res, all_tr_traces, all_te_traces

    def _train_and_evaluate_variant(self, variant_name, config, train_t1, train_t2, train_y, train_raw, test_t1, test_t2, test_y, test_raw, emb_dim, seed):
        """Worker function to train and evaluate a single BPTT variant independently."""
        torch.manual_seed(seed)
        
        # 1. Initialize Centroids and Model
        centroids = torch.zeros(3, emb_dim, device=self.device)
        for i in range(3):
            mask = (train_y == i)
            centroids[i] = train_t2[mask].mean(dim=0) if mask.sum() > 0 else torch.randn(emb_dim, device=self.device)
                
        model = NeuroSymbolicBeliefOptimizer(embedding_dim=emb_dim, num_classes=3, initial_vectors=centroids).to(self.device)
        criterion = NSBRLoss(config, beta=config.get("beta", 0.1))
        optimizer = optim.AdamW(model.parameters(), lr=0.015, weight_decay=0.1)

        # 2. Execute PyTorch Training Loop (1000 Epochs)
        model.train()
        pbar = tqdm(range(1000), desc=f"BPTT [{variant_name}]", leave=False)
        for epoch in pbar:
            optimizer.zero_grad()
            _, v_norm = model(train_t1)
            t1_raw_sims = torch.matmul(F.normalize(train_t1, p=2, dim=1), v_norm.T)
            t2_raw_sims = torch.matmul(F.normalize(train_t2, p=2, dim=1), v_norm.T)
            loss = criterion(t1_raw_sims, t2_raw_sims, train_y)
            loss.backward()
            optimizer.step()
            
            if epoch % 100 == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        # 3. Evaluate the Learned Vectors
        model.eval()
        with torch.no_grad():
            _, v_norm = model(train_t1)
            t1_sim_train = torch.matmul(F.normalize(train_t1, p=2, dim=1), v_norm.T)
            t2_sim_train = torch.matmul(F.normalize(train_t2, p=2, dim=1), v_norm.T)
            t1_sim_test = torch.matmul(F.normalize(test_t1, p=2, dim=1), v_norm.T)
            t2_sim_test = torch.matmul(F.normalize(test_t2, p=2, dim=1), v_norm.T)
            
            target_gate = config.get("eval_gate", 0.66) 
            target_sim = config.get("min_sim", 0.25) 
            
            tr_qpr, tr_drr, tr_traces = self._evaluate_universal_bayesian(
                t1_sim_train, t2_sim_train, train_y, target_gate, target_sim, 
                beta=config.get("beta", 0.1), raw_items=train_raw, method_name=variant_name
            )
            te_qpr, te_drr, te_traces = self._evaluate_universal_bayesian(
                t1_sim_test, t2_sim_test, test_y, target_gate, target_sim, 
                beta=config.get("beta", 0.1), raw_items=test_raw, method_name=variant_name
            )
            
        return variant_name, tr_qpr, tr_drr, te_qpr, te_drr, tr_traces, te_traces

    def run_learned_evaluation(self, train_t1, train_t2, train_y, train_raw, test_t1, test_t2, test_y, test_raw, emb_dim, seed):
        """Runs BPTT parameter optimization sequentially per variant with live feedback to prevent PyTorch deadlocks."""
        print(f"\n{'='*90}\nRUNNING LEARNED EVALUATIONS - SEED: {seed}\n{'='*90}")
        train_res, test_res, full_res = {}, {}, {}
        all_tr_traces, all_te_traces = [], []

        # Execute variants sequentially to avoid OpenMP/CUDA thread contention
        for variant_name, config in self.variants_config.items():
            print(f"\nOptimizing Variant: {variant_name}...")
            
            variant_name, tr_qpr, tr_drr, te_qpr, te_drr, tr_traces, te_traces = self._train_and_evaluate_variant(
                variant_name, config, train_t1, train_t2, train_y, train_raw, 
                test_t1, test_t2, test_y, test_raw, emb_dim, seed
            )
            
            all_tr_traces.extend(tr_traces)
            all_te_traces.extend(te_traces)

            print(f"Results for {variant_name}: Train [DPR: {tr_qpr:.1f}% | DRR: {tr_drr:.1f}%] || Test [DPR: {te_qpr:.1f}% | DRR: {te_drr:.1f}%]")

            display = f"Learned:\n{variant_name.replace('BPTT_', '')}"
            train_res[display] = {"QPR": tr_qpr, "DRR": tr_drr}
            test_res[display] = {"QPR": te_qpr, "DRR": te_drr}
            full_res[display] = {"QPR": te_qpr, "DRR": te_drr}

        return train_res, test_res, full_res, all_tr_traces, all_te_traces

    def execute_monte_carlo(self, seeds=[42, 123, 777, 2026, 9999]):
        """Executes the pipeline across multiple seeds and aggregates results and traces."""
        agg_train, agg_test, agg_full = {}, {}, {}
        master_train_traces, master_test_traces = [], []

        for seed in seeds:
            print(f"\n>> EXECUTING MONTE CARLO SEED: {seed}")
            train_t1, train_t2, train_y, train_raw, test_t1, test_t2, test_y, test_raw, emb_dim = self.prepare_data(seed)
            
            t_llm, te_llm, f_llm, llm_tr_traces, llm_te_traces = self.run_llm_baseline(train_raw, test_raw)
            t_man, te_man, f_man, man_tr_traces, man_te_traces = self.run_manual_evaluation(train_t1, train_t2, train_y, train_raw, test_t1, test_t2, test_y, test_raw, seed)
            t_lrn, te_lrn, f_lrn, lrn_tr_traces, lrn_te_traces = self.run_learned_evaluation(train_t1, train_t2, train_y, train_raw, test_t1, test_t2, test_y, test_raw, emb_dim, seed)
            
            master_train_traces.extend(llm_tr_traces + man_tr_traces + lrn_tr_traces)
            master_test_traces.extend(llm_te_traces + man_te_traces + lrn_te_traces)
            
            cur_train = {**t_llm, **t_man, **t_lrn}
            cur_test = {**te_llm, **te_man, **te_lrn}
            cur_full = {**f_llm, **f_man, **f_lrn}
            
            for key in cur_train:
                if key not in agg_train:
                    agg_train[key] = {"QPR": [], "DRR": []}
                    agg_test[key]  = {"QPR": [], "DRR": []}
                    agg_full[key]  = {"QPR": [], "DRR": []}
                    
                agg_train[key]["QPR"].append(cur_train[key]["QPR"])
                agg_train[key]["DRR"].append(cur_train[key]["DRR"])
                agg_test[key]["QPR"].append(cur_test[key]["QPR"])
                agg_test[key]["DRR"].append(cur_test[key]["DRR"])
                agg_full[key]["QPR"].append(cur_full[key]["QPR"])
                agg_full[key]["DRR"].append(cur_full[key]["DRR"])

        train_trace_path = os.path.join("results", f"{self.phase_prefix}_train_traces_master.json")
        test_trace_path = os.path.join("results", f"{self.phase_prefix}_test_traces_master.json")
        with open(train_trace_path, "w", encoding="utf-8") as f:
            json.dump(master_train_traces, f, indent=4)
        with open(test_trace_path, "w", encoding="utf-8") as f:
            json.dump(master_test_traces, f, indent=4)

        final_train, final_test, final_full = {}, {}, {}
        print(f"\n{'='*115}")
        print(f"FINAL AGGREGATED RESULTS OVER {len(seeds)} SEEDS (MEAN ± STD DEV)")
        print(f"{'='*115}")
        print(f"{'ARCHITECTURE / VARIANT':<30} | {'TRAIN DPR':<16} | {'TRAIN DRR':<16} || {'TEST DPR':<16} | {'TEST DRR':<16}")
        print("-" * 115)
        
        for key in agg_test:
            final_train[key] = {"QPR": np.mean(agg_train[key]["QPR"]), "DRR": np.mean(agg_train[key]["DRR"])}
            final_test[key]  = {"QPR": np.mean(agg_test[key]["QPR"]),  "DRR": np.mean(agg_test[key]["DRR"])}
            final_full[key]  = {"QPR": np.mean(agg_full[key]["QPR"]),  "DRR": np.mean(agg_full[key]["DRR"])}
            
            # Train Metrics
            tr_dpr_m, tr_dpr_s = np.mean(agg_train[key]["QPR"]), np.std(agg_train[key]["QPR"])
            tr_drr_m, tr_drr_s = np.mean(agg_train[key]["DRR"]), np.std(agg_train[key]["DRR"])

            # Test Metrics
            te_dpr_m, te_dpr_s = np.mean(agg_test[key]["QPR"]), np.std(agg_test[key]["QPR"])
            te_drr_m, te_drr_s = np.mean(agg_test[key]["DRR"]), np.std(agg_test[key]["DRR"])
            
            clean_key = key.replace('\n', ' ')
            print(f"{clean_key:<30} | {tr_dpr_m:>5.1f}% ± {tr_dpr_s:>3.1f}% | {tr_drr_m:>5.1f}% ± {tr_drr_s:>3.1f}% || {te_dpr_m:>5.1f}% ± {te_dpr_s:>3.1f}% | {te_drr_m:>5.1f}% ± {te_drr_s:>3.1f}%")

        return final_train, final_test, final_full, train_trace_path, test_trace_path
import os
import sys
import json
import time
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
from collections import defaultdict
from sklearn.model_selection import train_test_split
from nsbr_core import NeuroSymbolicBeliefOptimizer, NSBRLoss
from ml_models import MultinomialLogisticRegression, SmallMLP
from concurrent.futures import ThreadPoolExecutor, as_completed
from agents import embed, KNOWLEDGE_BASE_TIERS, LLMBaselineAgent, FewShotLLMBaselineAgent

class ExperimentPipeline:
    """
    Executes the multi-stage evaluation pipeline for Neuro-Symbolic Belief Routing.
    Handles data stratification, zero-shot and dynamic few-shot LLM baseline testing, 
    manual geometric checks, and BPTT learned parameter optimization.
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

    METRICS_LIST = ["DPR", "DRR", "SAR", "FAR", "TPP", "TP", "FP", "TN", "FN"]

    def __init__(self, json_path, device, phase_prefix, variants_config, model_id="anthropic.claude-sonnet-4-6"):
        self.json_path = json_path
        self.device = device
        self.phase_prefix = phase_prefix
        self.variants_config = variants_config
        self.model_id = model_id

    @staticmethod
    def _build_exemplars_from_train(train_raw, k_shots, seed=42):
        """Extracts k balanced few-shot exemplars directly from the seed's training split."""
        if k_shots == 0:
            return []
        
        rng = np.random.RandomState(seed)
        
        categories = {
            "dvla_direct": [],
            "dvsa_direct": [],
            "ambiguous_id": [],
            "ood_trap": [],
            "far_ood": []
        }
        
        for item in train_raw:
            gt = item.get("ground_truth", "").upper()
            tier = item.get("operational_tier", "")
            
            if tier == "Ambiguous_ID":
                categories["ambiguous_id"].append(item)
            elif "DVLA" in gt:
                categories["dvla_direct"].append(item)
            elif "DVSA" in gt:
                categories["dvsa_direct"].append(item)
            elif tier == "OOD_Trap" or "near" in item.get("id", "").lower():
                categories["ood_trap"].append(item)
            else:
                categories["far_ood"].append(item)
        
        for cat in categories:
            rng.shuffle(categories[cat])
            
        exemplars = []
        cat_order = ["dvla_direct", "dvsa_direct", "ambiguous_id", "ood_trap", "far_ood"]
        cat_idx = 0
        
        while len(exemplars) < k_shots:
            cat_key = cat_order[cat_idx % len(cat_order)]
            cat_idx += 1
            
            if categories[cat_key]:
                item = categories[cat_key].pop(0)
                prompt_text = item["dialogue"][0]
                gt = item["ground_truth"].upper()
                tier = item["operational_tier"]
                
                if tier == "Ambiguous_ID":
                    action = "abstain"
                elif "DVLA" in gt:
                    action = "tool_dvla"
                elif "DVSA" in gt:
                    action = "tool_dvsa"
                else:
                    action = "refuse"
                    
                exemplars.append({"user": prompt_text, "action": action})
                
            if all(len(v) == 0 for v in categories.values()):
                break
                
        return exemplars

    @staticmethod
    def _calculate_derived_metrics(tp, fp, tn, fn, total, in_domain_total, ood_total):
        """Unified metric calculation for all evaluation types."""
        dpr = ((tp + tn + fn) / total) * 100 if total > 0 else 0.0
        drr = (tp / in_domain_total) * 100 if in_domain_total > 0 else 0.0
        sar = (tn / ood_total) * 100 if ood_total > 0 else 0.0
        far = (fn / in_domain_total) * 100 if in_domain_total > 0 else 0.0
        tpp = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0.0
        
        return {
            "DPR": dpr, "DRR": drr, "SAR": sar, "FAR": far, "TPP": tpp,
            "TP": tp, "FP": fp, "TN": tn, "FN": fn
        }

    @staticmethod
    def _save_json(data, filepath):
        """Standardized JSON saving with automatic directory creation."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
            
    @staticmethod
    def _save_csv(results_dict, filepath):
        """Standardized CSV export for evaluation metrics."""
        records = []
        for method_key, metrics in results_dict.items():
            row = {"Method": method_key.replace('\n', ' ')}
            for m_name, val in metrics.items():
                if m_name in ["TP", "FP", "TN", "FN"]:
                    row[m_name] = int(round(val))
                else:
                    row[m_name] = round(val, 2)
            records.append(row)
        
        df = pd.DataFrame(records)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        print(f"Saved metrics summary CSV to: '{filepath}'")
    def _log_and_store_results(self, variant_name, display_name, tr_metrics, te_metrics, tr_traces, te_traces, 
                               all_tr_traces, all_te_traces, train_res, test_res, full_res):
        """Handles the repetitive tracing, storing, and printing of evaluation loops."""
        all_tr_traces.extend(tr_traces)
        all_te_traces.extend(te_traces)
        
        train_res[display_name] = tr_metrics
        test_res[display_name] = te_metrics
        full_res[display_name] = te_metrics 

        print(f"{variant_name:<30} | {tr_metrics['DPR']:>8.1f}% | {tr_metrics['DRR']:>8.1f}% || {te_metrics['DPR']:>8.1f}% | {te_metrics['DRR']:>8.1f}%")

    def _compute_splits(self, raw_data, stratify_labels, seed):
        """Determines train/test split indices, handling Grouped Stratified Splits if necessary."""
        train_idx, test_idx = [], []
        has_groups = any("base_intent_id" in item for item in raw_data)
        
        if has_groups:
            print("Grouping Key Detected: Performing Stratified Intent Group Split...")
            class_to_indices = {0: [], 1: [], 2: [], 3: []}
            for i, label in enumerate(stratify_labels):
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
            indices = np.arange(len(raw_data))
            train_idx, test_idx = train_test_split(
                indices, test_size=0.2, stratify=stratify_labels, random_state=seed
            )
            
        train_idx, test_idx = np.array(train_idx), np.array(test_idx)
        print(f"Data Split Complete: {len(train_idx)} Train, {len(test_idx)} Test")
        return train_idx, test_idx

    def _get_embeddings(self, texts_t1, texts_t2):
        """Loads cached embeddings or generates them concurrently via ThreadPoolExecutor."""
        cache_file = os.path.join("cache", f"{self.phase_prefix}_titan_10k_embeddings_cache.pt")
        os.makedirs("cache", exist_ok=True)
        os.makedirs("results", exist_ok=True)

        if os.path.exists(cache_file):
            print(f"FAST LOAD: Found cached Titan V2 embeddings at '{cache_file}'.")
            cache = torch.load(cache_file, weights_only=True)
            all_t1_embs = cache['t1'].to(self.device)
            all_t2_embs = cache['t2'].to(self.device)
            return all_t1_embs, all_t2_embs

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
        
        return all_t1_embs.to(self.device), all_t2_embs.to(self.device)

    def _get_llm_traces(self, dataset, cache_dict, method_label):
        """Generates standard trace dictionaries from the LLM cache."""
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
                    "method": method_label,
                    "user_sample": {
                        "turn_1": dialogue[0] if len(dialogue) > 0 else "",
                        "turn_2": dialogue[1] if len(dialogue) > 1 else (dialogue[0] if len(dialogue) > 0 else "")
                    },
                    "llm_interaction": cached_data.get("history", []),
                    "turns_taken": cached_data.get("turns_taken", 0),
                    "outcome": cached_data["outcome"]
                })
        return traces

    def _calc_llm_metrics(self, dataset, cache_dict):
        """Calculates derived metrics (DPR, DRR, etc.) from LLM outcomes."""
        if not dataset: 
            return self._calculate_derived_metrics(0, 0, 0, 0, 0, 0, 0)
            
        total = len(dataset)
        ood_total = sum(1 for item in dataset if item["ground_truth"].upper() == "OOD")
        in_domain_total = total - ood_total
        
        outcomes = [cache_dict[item["id"]]["outcome"] for item in dataset if item["id"] in cache_dict]
        
        tp = outcomes.count("SR")
        tn = outcomes.count("TN")
        fn = outcomes.count("SA")  # Safe Abstention on valid query = False Abstention
        fp = outcomes.count("MR")  # Misroute = Safety/Precision Failure
        
        return self._calculate_derived_metrics(tp, fp, tn, fn, total, in_domain_total, ood_total)

    def _execute_llm_inference(self, missing_items, shots, exemplars, cache_dict, cache_file, cache_tag):
        """Processes missing LLM inferences concurrently with a ThreadPoolExecutor."""
        fatal_error = None
        completed_count = 0
        
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = {
                executor.submit(self._evaluate_single_llm_trajectory, item, shots, exemplars): item 
                for item in missing_items
            }
            
            for future in tqdm(as_completed(futures), total=len(missing_items), desc=f"LLM {cache_tag} Inference", leave=False):
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
                    # Save intermediate progress to avoid data loss
                    if completed_count % 100 == 0:
                        self._save_json(cache_dict, cache_file)

        # Final save for the batch
        self._save_json(cache_dict, cache_file)

        if fatal_error:
            print(f"\n{'!'*90}")
            print("AWS CREDENTIAL EXPIRED OR FATAL CONNECTION ERROR ENCOUNTERED!")
            print(f"Details: {fatal_error}")
            print(f"{'!'*90}\n")
            sys.exit(1)
            
        return cache_dict

    def _execute_single_run(self, seed):
            """Executes a single Monte Carlo run for a given seed."""
            print(f"\n>> EXECUTING MONTE CARLO SEED: {seed}")
            
            # 1. Prepare Data
            train_t1, train_t2, train_y, train_raw, test_t1, test_t2, test_y, test_raw, emb_dim = self.prepare_data(seed)
            
            # 2. Save Splits
            train_split_path = os.path.join("data", "splits", f"{self.phase_prefix}_train_seed_{seed}.json")
            test_split_path = os.path.join("data", "splits", f"{self.phase_prefix}_test_seed_{seed}.json")
            
            self._save_json(train_raw, train_split_path)
            self._save_json(test_raw, test_split_path)
            print(f"Saved Train/Test splits for Seed {seed} to data/splits/")
            
            # 3. Run Evaluations
            t_llm, te_llm, f_llm, llm_tr_traces, llm_te_traces = self.run_llm_baseline(train_raw, test_raw, seed=seed)
            t_man, te_man, f_man, man_tr_traces, man_te_traces = self.run_nsbr_heuristic(
                train_t1, train_t2, train_y, train_raw, test_t1, test_t2, test_y, test_raw, seed
            )
            t_cls, te_cls, f_cls, cls_tr_traces, cls_te_traces = self.run_nsbr_classical_ml(
                train_t1, train_t2, train_y, train_raw, test_t1, test_t2, test_y, test_raw, emb_dim, seed
            )
            t_lrn, te_lrn, f_lrn, lrn_tr_traces, lrn_te_traces = self.run_nsbr_data_driven_evaluation(
                train_t1, train_t2, train_y, train_raw, test_t1, test_t2, test_y, test_raw, emb_dim, seed
            )
            
            # 4. Merge Results
            cur_train = {**t_llm, **t_man, **t_cls, **t_lrn}
            cur_test = {**te_llm, **te_man, **te_cls, **te_lrn}
            cur_full = {**f_llm, **f_man, **f_cls, **f_lrn}
            
            run_train_traces = llm_tr_traces + man_tr_traces + cls_tr_traces + lrn_tr_traces
            run_test_traces = llm_te_traces + man_te_traces + cls_te_traces + lrn_te_traces
            
            return cur_train, cur_test, cur_full, run_train_traces, run_test_traces
    
    def _aggregate_and_save_metrics(self, agg_train, agg_test, agg_full, num_seeds):
        """Calculates statistics across seeds, prints the summary table, and exports CSVs."""
        final_train, final_test, final_full = {}, {}, {}
        
        print(f"\n{'='*140}")
        print(f"FINAL AGGREGATED RESULTS OVER {num_seeds} SEEDS (TEST SET SUMMARY)")
        print(f"{'='*140}")
        print(f"{'ARCHITECTURE / VARIANT':<30} | {'DPR (Safety)':<18} | {'DRR (Yield)':<18} | {'SAR (Defense)':<18} | {'FAR (Friction)':<18}")
        print("-" * 140)
        
        
        for key in agg_test:
            # Calculate means
            final_train[key] = {m: float(np.mean(agg_train[key][m])) for m in self.METRICS_LIST}
            final_test[key]  = {m: float(np.mean(agg_test[key][m])) for m in self.METRICS_LIST}
            final_full[key]  = {m: float(np.mean(agg_full[key][m])) for m in self.METRICS_LIST}
            
            # Calculate standard deviations for display
            te_dpr_m, te_dpr_s = final_test[key]["DPR"], float(np.std(agg_test[key]["DPR"]))
            te_drr_m, te_drr_s = final_test[key]["DRR"], float(np.std(agg_test[key]["DRR"]))
            te_sar_m, te_sar_s = final_test[key]["SAR"], float(np.std(agg_test[key]["SAR"]))
            te_far_m, te_far_s = final_test[key]["FAR"], float(np.std(agg_test[key]["FAR"]))
            
            clean_key = key.replace('\n', ' ')
            
            # Format display strings
            dpr_str = f"{te_dpr_m:>5.1f}% ± {te_dpr_s:>3.1f}%"
            drr_str = f"{te_drr_m:>5.1f}% ± {te_drr_s:>3.1f}%"
            sar_str = f"{te_sar_m:>5.1f}% ± {te_sar_s:>3.1f}%"
            far_str = f"{te_far_m:>5.1f}% ± {te_far_s:>3.1f}%"
            
            print(f"{clean_key:<30} | {dpr_str:<18} | {drr_str:<18} | {sar_str:<18} | {far_str:<18}")

        # Save CSVs
        self._save_csv(final_train, os.path.join("results", f"{self.phase_prefix}_train_metrics_summary.csv"))
        self._save_csv(final_test, os.path.join("results", f"{self.phase_prefix}_test_metrics_summary.csv"))
        self._save_csv(final_full, os.path.join("results", f"{self.phase_prefix}_full_metrics_summary.csv"))
        
        return final_train, final_test, final_full

    def _train_static_classifier(self, model, train_x, train_y, epochs=300, lr=0.01):
        """Trains a static ML baseline classifier on conversational representations."""
        model = model.to(self.device)
        
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        criterion = torch.nn.CrossEntropyLoss()
        
        model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            logits = model(train_x)
            loss = criterion(logits, train_y)
            loss.backward()
            optimizer.step()
            
        model.eval()
        return model

    def _evaluate_single_llm_trajectory(self, item, shots=0, exemplars=None, max_retries=5):
        gt_intent = item["ground_truth"]
        dialogue_turns = item["dialogue"]
        
        if shots == 0:
            agent = LLMBaselineAgent(model_name=self.model_id, aws_region="eu-west-2")
        else:
            agent = FewShotLLMBaselineAgent(model_name=self.model_id, aws_region="eu-west-2", exemplars=exemplars, shots=shots)
        
        outcome = "SA"
        conversation_context = ""
        last_res = ""
        turns_taken = 0
        trajectory_history = []
        
        for turn in dialogue_turns:
            turns_taken += 1
            conversation_context += f"{turn}\n"
            attempts = 0
            
            while True:
                attempts += 1
                try:
                    last_res = agent.send_message(conversation_context)
                    break
                except Exception as exc:
                    exc_str = str(exc)
                    if any(err_kw in exc_str for err_kw in [
                        "ExpiredToken", "Credentials", "expired", "UnrecognizedClientException", 
                        "AccessDenied", "EndpointConnectionError"
                    ]):
                        raise RuntimeError(f"FATAL_AWS_CREDENTIAL_ERROR: {exc_str}")

                    if attempts <= max_retries and any(t_kw in exc_str for t_kw in ["ThrottlingException", "Too many requests", "RateExceeded"]):
                        sleep_time = (2 ** attempts) * 0.5
                        time.sleep(sleep_time)
                        continue
                    else:
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

            trajectory_history.append({
                "turn": turns_taken,
                "user_prompt": turn,
                "llm_response": last_res
            })

            res_clean = last_res.lower()
            if "decision executed" in res_clean or "connect_to" in res_clean:
                gt_clean = gt_intent.lower().replace("connect_to_", "").strip()
                if gt_intent.upper() == "OOD" or gt_clean == "ood":
                    outcome = "MR"
                elif gt_clean in res_clean:
                    outcome = "SR"
                else:
                    outcome = "MR"
                break
            else:
                conversation_context += f"Assistant: {last_res}\n"
                
        if outcome == "SA" and (gt_intent.upper() == "OOD" or gt_intent == "ood"):
            outcome = "TN"
            
        return item["id"], {
            "history": trajectory_history,
            "outcome": outcome,
            "turns_taken": turns_taken
        }

    def run_llm_baseline(self, train_raw, test_raw, seed=42):
        """Executes Zero-Shot, 2-Shot, 5-Shot, and 10-Shot LLM evaluations."""
        print(f"\n{'='*90}\nRUNNING LLM BASELINES (Seed: {seed}) - ({self.phase_prefix})\n{'='*90}")
        full_raw = train_raw + test_raw
        
        shot_configs = [
            (0, "Zero-Shot:\nLLM Baseline", "0shot"),
            (2, "2-Shot:\nLLM Baseline", "2shot"),
            (5, "5-Shot:\nLLM Baseline", "5shot"),
            (10, "10-Shot:\nLLM Baseline", "10shot")
        ]
        
        train_res_all, test_res_all, full_res_all = {}, {}, {}
        all_train_traces, all_test_traces = [], []

        print(f"{'VARIANT NAME':<20} | {'TRAIN DPR':<9} | {'TRAIN DRR':<9} || {'TEST DPR':<9} | {'TEST DRR':<9}")
        print("-" * 75)

        for shots, display_name, cache_tag in shot_configs:
            cache_file = os.path.join("cache", f"{self.phase_prefix}_llm_{cache_tag}_seed_{seed}_cache.json")
            cache_dict = {}
            
            # 1. Setup Exemplars
            exemplars = self._build_exemplars_from_train(train_raw, shots, seed=seed)
            if shots > 0:
                self._save_json(exemplars, os.path.join("data", "splits", f"{self.phase_prefix}_exemplars_{cache_tag}_seed_{seed}.json"))

            # 2. Load and clean cache
            if os.path.exists(cache_file):
                with open(cache_file, "r", encoding="utf-8") as f:
                    cache_dict = json.load(f)
                purged_keys = [k for k, v in cache_dict.items() if "ERROR:" in str(v.get("llm_response", ""))]
                for k in purged_keys:
                    del cache_dict[k]

            # 3. Process missing items
            missing = [item for item in full_raw if item["id"] not in cache_dict]
            if missing:
                cache_dict = self._execute_llm_inference(missing, shots, exemplars, cache_dict, cache_file, cache_tag)
                    
            # 4. Generate Traces
            method_label = display_name.replace("\n", " ")
            all_train_traces.extend(self._get_llm_traces(train_raw, cache_dict, method_label))
            all_test_traces.extend(self._get_llm_traces(test_raw, cache_dict, method_label))
                    
            # 5. Calculate Metrics
            tr_metrics = self._calc_llm_metrics(train_raw, cache_dict)
            te_metrics = self._calc_llm_metrics(test_raw, cache_dict)
            f_metrics = self._calc_llm_metrics(full_raw, cache_dict)
            
            print(f"{method_label:<20} | {tr_metrics['DPR']:>8.1f}% | {tr_metrics['DRR']:>8.1f}% || {te_metrics['DPR']:>8.1f}% | {te_metrics['DRR']:>8.1f}%")

            train_res_all[display_name] = tr_metrics
            test_res_all[display_name]  = te_metrics
            full_res_all[display_name]  = f_metrics

        return train_res_all, test_res_all, full_res_all, all_train_traces, all_test_traces

    @staticmethod
    def _evaluate_gated_routing(t1_raw_sim, t2_raw_sim, targets, gate_threshold, min_sim_threshold, beta=0.1, lambda_decay=0.8, raw_items=None, method_name=""):
        """Computes deterministic safety and yield metrics over trajectory tensors and returns deep traces."""
        
        # 1. Inner helper to DRY up the math
        def get_stats(raw_sim, prior):
            unnorm = torch.exp(raw_sim / beta) * (prior ** lambda_decay)
            probs = unnorm / torch.sum(unnorm, dim=-1, keepdim=True)
            ent = -torch.sum(probs * torch.log2(probs + 1e-9), dim=-1)
            preds = torch.argmax(probs, dim=-1)
            max_sim, _ = torch.max(raw_sim, dim=-1)
            appr = (ent <= gate_threshold) & (max_sim >= min_sim_threshold)
            return probs, ent, preds, max_sim, appr

        # Calculate Turn 1 and Turn 2 stats
        t1_probs, t1_ent, t1_preds, t1_max, t1_appr = get_stats(t1_raw_sim, torch.ones_like(t1_raw_sim) / 3.0)
        t2_probs, t2_ent, t2_preds, t2_max, t2_appr = get_stats(t2_raw_sim, t1_probs)
        
        t1_acts = t1_appr & (t1_preds != 2)
        t2_acts = (~t1_acts) & t2_appr & (t2_preds != 2)
        any_acts = t1_acts | t2_acts
        
        # 2. Condensed Metric Logic
        is_in_domain, is_ood = (targets != 2), (targets == 2)
        
        tp = (((t1_acts & (t1_preds == targets)) | (t2_acts & (t2_preds == targets))) & is_in_domain).sum().item()
        fp_wrong = (((t1_acts & (t1_preds != targets)) | (t2_acts & (t2_preds != targets))) & is_in_domain).sum().item()
        fp = fp_wrong + (any_acts & is_ood).sum().item()
        tn = (~any_acts & is_ood).sum().item()
        fn = (~any_acts & is_in_domain).sum().item()
        
        metrics_dict = ExperimentPipeline._calculate_derived_metrics(
            tp, fp, tn, fn, len(targets), is_in_domain.sum().item(), is_ood.sum().item()
        )

        # 3. Clean Zipped Trace Generation
        traces = []
        if raw_items:
            zipped_data = zip(
                raw_items, targets.tolist(), t1_acts.tolist(), t2_acts.tolist(),
                t1_preds.tolist(), t2_preds.tolist(), t1_probs.tolist(), t2_probs.tolist(),
                t1_ent.tolist(), t2_ent.tolist(), t1_max.tolist(), t2_max.tolist()
            )
            
            for item, gt, a1, a2, p1, p2, pr1, pr2, e1, e2, m1, m2 in zipped_data:
                pred, turn = (p1, 1) if a1 else ((p2, 2) if a2 else (-1, 0))
                
                if pred == gt and gt != 2: outcome = "SR"
                elif pred == -1 and gt == 2: outcome = "TN"
                elif pred == -1 and gt != 2: outcome = "SA"
                else: outcome = "MR"

                d = item.get("dialogue", [])
                t1_text = d[0] if len(d) > 0 else ""
                t2_text = d[1] if len(d) > 1 else t1_text

                traces.append({
                    "trajectory_id": item["id"],
                    "base_intent_id": item.get("base_intent_id", item["id"]),
                    "operational_tier": item["operational_tier"],
                    "ground_truth": item["ground_truth"],
                    "method": method_name,
                    "user_sample": {"turn_1": t1_text, "turn_2": t2_text},
                    "t1_probs": [round(p, 4) for p in pr1], "t1_entropy": round(e1, 4), "t1_max_sim": round(m1, 4),
                    "t2_probs": [round(p, 4) for p in pr2], "t2_entropy": round(e2, 4), "t2_max_sim": round(m2, 4),
                    "routed_turn": turn, "prediction_class": pred, "outcome": outcome
                })

        return metrics_dict, traces

    def label_and_process(self, corpus):
            """Labels each trajectory with its operational tier and prepares for embedding."""
            texts_t1, texts_t2, raw_data = [], [], []
            labels_3way, stratify_labels_4way = [], []
            
            for item in corpus:
                traj_id  = item.get("id", "")
                gt       = item.get("ground_truth", "")
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
                    
            return texts_t1, texts_t2, raw_data, labels_3way, stratify_labels_4way

    def prepare_data(self, seed):

        """Orchestrates data loading, splitting, and embedding generation."""
        print(f"\nLoading corpus from {self.json_path} (Seed: {seed})...")
        with open(self.json_path, 'r', encoding='utf-8') as f:
            corpus = json.load(f)

        # 1. Label and Process 
        texts_t1, texts_t2, raw_data, labels_3way, stratify_labels_4way = self.label_and_process(corpus)
                    
        # 2. Determine Train/Test Splits
        train_idx, test_idx = self._compute_splits(raw_data, stratify_labels_4way, seed)

        # 3. Retrieve or Generate Embeddings
        all_t1_embs, all_t2_embs = self._get_embeddings(texts_t1, texts_t2)
        
        # 4. Finalize Tensors
        all_labels = torch.tensor(labels_3way, dtype=torch.long).to(self.device)
        
        train_t1, train_t2, train_y = all_t1_embs[train_idx], all_t2_embs[train_idx], all_labels[train_idx]
        train_raw = [raw_data[i] for i in train_idx]
        
        test_t1, test_t2, test_y = all_t1_embs[test_idx], all_t2_embs[test_idx], all_labels[test_idx]
        test_raw = [raw_data[i] for i in test_idx]
        
        return train_t1, train_t2, train_y, train_raw, test_t1, test_t2, test_y, test_raw, all_t1_embs.shape[1]

    def run_nsbr_data_driven(self, variant_name, config, train_t1, train_t2, train_y, train_raw, test_t1, test_t2, test_y, test_raw, emb_dim, seed):
        """Worker function to train and evaluate a single NSBR (Data-Driven) variant independently."""
        torch.manual_seed(seed)
        
        # 1. Pre-normalize static inputs (Massive speedup: avoids re-normalizing 1,000 times)
        train_t1_norm = F.normalize(train_t1, p=2, dim=1)
        train_t2_norm = F.normalize(train_t2, p=2, dim=1)
        test_t1_norm = F.normalize(test_t1, p=2, dim=1)
        test_t2_norm = F.normalize(test_t2, p=2, dim=1)
        
        # 2. Initialize Centroids
        centroids = torch.zeros(3, emb_dim, device=self.device)
        for i in range(3):
            mask = (train_y == i)
            centroids[i] = train_t2[mask].mean(dim=0) if mask.sum() > 0 else torch.randn(emb_dim, device=self.device)
                
        # 3. Model Setup
        model = NeuroSymbolicBeliefOptimizer(embedding_dim=emb_dim, num_classes=3, initial_vectors=centroids).to(self.device)
        criterion = NSBRLoss(config, beta=config.get("beta", 0.1))
        optimizer = optim.AdamW(model.parameters(), lr=0.015, weight_decay=0.1)

        # 4. Training Loop
        model.train()
        pbar = tqdm(range(1000), desc=f"NSBR Data-Driven [{variant_name}]", leave=False)
        for epoch in pbar:
            optimizer.zero_grad()
            _, v_norm = model(train_t1) 
            
            # Use the pre-normalized tensors for lightning-fast matrix multiplication
            t1_raw_sims = torch.matmul(train_t1_norm, v_norm.T)
            t2_raw_sims = torch.matmul(train_t2_norm, v_norm.T)
            
            loss = criterion(t1_raw_sims, t2_raw_sims, train_y)
            loss.backward()
            optimizer.step()
            
            if epoch % 100 == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        # 5. Evaluation Phase
        model.eval()
        with torch.no_grad():
            _, v_norm = model(train_t1)
            t1_sim_train = torch.matmul(train_t1_norm, v_norm.T)
            t2_sim_train = torch.matmul(train_t2_norm, v_norm.T)
            t1_sim_test = torch.matmul(test_t1_norm, v_norm.T)
            t2_sim_test = torch.matmul(test_t2_norm, v_norm.T)
            
            target_gate = config.get("eval_gate", 0.66) 
            target_sim = config.get("eval_sim", config.get("min_sim", 0.25))
            
            tr_metrics, tr_traces = self._evaluate_gated_routing(
                t1_sim_train, t2_sim_train, train_y, target_gate, target_sim, 
                beta=config.get("beta", 0.1), raw_items=train_raw, method_name=variant_name
            )
            te_metrics, te_traces = self._evaluate_gated_routing(
                t1_sim_test, t2_sim_test, test_y, target_gate, target_sim, 
                beta=config.get("beta", 0.1), raw_items=test_raw, method_name=variant_name
            )
            
        return variant_name, tr_metrics, te_metrics, tr_traces, te_traces

    def run_nsbr_heuristic(self, train_t1, train_t2, train_y, train_raw, test_t1, test_t2, test_y, test_raw, seed):
        """Runs geometric checks and builds train and test traces."""
        print(f"\n{'='*90}\nRUNNING MANUAL EVALUATIONS (TITAN V2 + BAYESIAN MATH)\n{'='*90}")
        train_res, test_res, full_res = {}, {}, {}
        all_tr_traces, all_te_traces = [], []
        t1_norm_test, t2_norm_test = F.normalize(test_t1, p=2, dim=1), F.normalize(test_t2, p=2, dim=1)
        t1_norm_train, t2_norm_train = F.normalize(train_t1, p=2, dim=1), F.normalize(train_t2, p=2, dim=1)
        
        print(f"{'TIER NAME':<20} | {'TRAIN DPR':<9} | {'TRAIN DRR':<9} || {'TEST DPR':<9} | {'TEST DRR':<9}")
        print("-" * 75)

        for tier_name, gate_rules in self.OPTIMIZED_GATES.items():
            if tier_name not in KNOWLEDGE_BASE_TIERS: continue
            kb = KNOWLEDGE_BASE_TIERS[tier_name]
            dest_embs = torch.tensor(np.array([embed(kb["DVLA"]), embed(kb["DVSA"]), embed(kb["OOD"])]), dtype=torch.float32).to(self.device)
            dest_norm = F.normalize(dest_embs, p=2, dim=1)
            
            t1_sim_train, t2_sim_train = torch.matmul(t1_norm_train, dest_norm.T), torch.matmul(t2_norm_train, dest_norm.T)
            t1_sim_test, t2_sim_test = torch.matmul(t1_norm_test, dest_norm.T), torch.matmul(t2_norm_test, dest_norm.T)
            
            method_name = f"Handcrafted_{tier_name}"
            
            tr_metrics, tr_traces = self._evaluate_gated_routing(
                t1_sim_train, t2_sim_train, train_y, gate_rules["max_entropy"], gate_rules["min_sim"], 
                raw_items=train_raw, method_name=method_name
            )
            te_metrics, te_traces = self._evaluate_gated_routing(
                t1_sim_test, t2_sim_test, test_y, gate_rules["max_entropy"], gate_rules["min_sim"], 
                raw_items=test_raw, method_name=method_name
            )

            display = f"Handcrafted:\n{tier_name}"
            self._log_and_store_results(
                tier_name, display, tr_metrics, te_metrics, tr_traces, te_traces,
                all_tr_traces, all_te_traces, train_res, test_res, full_res
            )
            
        return train_res, test_res, full_res, all_tr_traces, all_te_traces

    def run_nsbr_classical_ml(self, train_t1, train_t2, train_y, train_raw, test_t1, test_t2, test_y, test_raw, emb_dim, seed):
        """Runs the classical ML baselines with mean-pooling and temperature scaling."""
        print(f"\n{'='*90}\nRUNNING CLASSICAL ML BASELINES (Seed: {seed})\n{'='*90}")
        train_res, test_res, full_res = {}, {}, {}
        all_tr_traces, all_te_traces = [], []

        # Mean-pool the multi-turn context
        train_pooled_t2 = (train_t1 + train_t2) / 2.0
        test_pooled_t2 = (test_t1 + test_t2) / 2.0

        models = {
            "Logistic Regression": MultinomialLogisticRegression(input_dim=emb_dim),
            "Small MLP": SmallMLP(input_dim=emb_dim)
        }

        # Temperature scalar sweep to test softmax overconfidence
        temperatures = [1.0, 2.0, 5.0, 10.0]

        print(f"{'VARIANT NAME':<30} | {'TRAIN DPR':<9} | {'TRAIN DRR':<9} || {'TEST DPR':<9} | {'TEST DRR':<9}")
        print("-" * 85)

        for model_name, model_instance in models.items():
            # Train the model on the fully contextualized Turn 2 representations
            trained_model = self._train_static_classifier(
                            model_instance, train_pooled_t2, train_y
                        )

            with torch.no_grad():
                # Generate logits for Turn 1 (instantaneous) and Turn 2 (pooled) once per model
                tr_logits_t1 = trained_model(train_t1)
                tr_logits_t2 = trained_model(train_pooled_t2)
                te_logits_t1 = trained_model(test_t1)
                te_logits_t2 = trained_model(test_pooled_t2)

                for temp in temperatures:
                    variant_name = f"Classical: {model_name} (T={temp})"
                    display_name = f"{model_name}\nT={temp}"
                    
                    # Pass logits as raw_sim, disable lambda decay (1.0), disable spatial gate (-999.0)
                    tr_metrics, tr_traces = self._evaluate_gated_routing(
                        tr_logits_t1, tr_logits_t2, train_y, 
                        gate_threshold=0.85, min_sim_threshold=-999.0, 
                        beta=temp, lambda_decay=1.0, 
                        raw_items=train_raw, method_name=variant_name
                    )
                    
                    te_metrics, te_traces = self._evaluate_gated_routing(
                        te_logits_t1, te_logits_t2, test_y, 
                        gate_threshold=0.85, min_sim_threshold=-999.0, 
                        beta=temp, lambda_decay=1.0, 
                        raw_items=test_raw, method_name=variant_name
                    )
                    
                    # Log and store inherently prints the formatted results
                    self._log_and_store_results(
                        variant_name, display_name, tr_metrics, te_metrics, tr_traces, te_traces,
                        all_tr_traces, all_te_traces, train_res, test_res, full_res
                    )

        return train_res, test_res, full_res, all_tr_traces, all_te_traces

    def run_nsbr_data_driven_evaluation(self, train_t1, train_t2, train_y, train_raw, test_t1, test_t2, test_y, test_raw, emb_dim, seed):
        """Runs BPTT parameter optimization sequentially per variant."""
        print(f"\n{'='*90}\nRUNNING LEARNED EVALUATIONS - SEED: {seed}\n{'='*90}")
        train_res, test_res, full_res = {}, {}, {}
        all_tr_traces, all_te_traces = [], []

        for variant_name, config in self.variants_config.items():
            print(f"\nOptimizing Variant: {variant_name}...")
            
            variant_name, tr_metrics, te_metrics, tr_traces, te_traces = self.run_nsbr_data_driven(
                variant_name, config, train_t1, train_t2, train_y, train_raw, 
                test_t1, test_t2, test_y, test_raw, emb_dim, seed
            )
            
            display = f"Learned:\n{variant_name.replace('BPTT_', '')}"
            self._log_and_store_results(
                variant_name, display, tr_metrics, te_metrics, tr_traces, te_traces,
                all_tr_traces, all_te_traces, train_res, test_res, full_res
            )
        return train_res, test_res, full_res, all_tr_traces, all_te_traces

    def execute_monte_carlo(self, seeds=[42, 123, 777, 2026, 9999]):
        """Orchestrates the pipeline across seeds, collects results/traces, and outputs summaries."""
        agg_train = defaultdict(lambda: defaultdict(list))
        agg_test = defaultdict(lambda: defaultdict(list))
        agg_full = defaultdict(lambda: defaultdict(list))
        master_train_traces, master_test_traces = [], []

        for seed in seeds:
            # 1. Execute single run
            cur_train, cur_test, cur_full, run_train_traces, run_test_traces = self._execute_single_run(seed)
            
            # 2. Append traces
            master_train_traces.extend(run_train_traces)
            master_test_traces.extend(run_test_traces)
            
            # 3. Collect metrics for aggregation
            for key in cur_train:
                for metric in ["DPR", "DRR", "SAR", "FAR", "TPP", "TP", "FP", "TN", "FN"]:
                    agg_train[key][metric].append(cur_train[key].get(metric, 0.0))
                    agg_test[key][metric].append(cur_test[key].get(metric, 0.0))
                    agg_full[key][metric].append(cur_full[key].get(metric, 0.0))

        # 4. Save Master Traces
        train_trace_path = os.path.join("results", f"{self.phase_prefix}_train_traces_master.json")
        test_trace_path = os.path.join("results", f"{self.phase_prefix}_test_traces_master.json")
        self._save_json(master_train_traces, train_trace_path)
        self._save_json(master_test_traces, test_trace_path)

        # 5. Aggregate, Print, and Export Summary Metrics
        final_train, final_test, final_full = self._aggregate_and_save_metrics(
            agg_train, agg_test, agg_full, len(seeds)
        )

        return final_train, final_test, final_full, train_trace_path, test_trace_path
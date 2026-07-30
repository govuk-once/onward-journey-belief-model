# Neuro-Symbolic Belief Routing (NSBR) Evaluation Framework

This repository contains the experimental evaluation pipeline for the **Neuro-Symbolic Belief Routing** framework. It validates a dual-process System 1 / System 2 routing architecture utilizing Titan V2 embeddings, Bayesian mathematics, and Backpropagation Through Time (BPTT).

## 📁 Repository Structure

Ensure the following three directories exist in the root of the project (the code will automatically scaffold them if missing):
*   `data/`: Stores the raw synthetic corpora and reference JSONs.
*   `cache/`: Stores the cached Titan V2 tensor embeddings to prevent redundant AWS API calls.
*   `results/`: Destination for all exported JSON traces, QPR/DRR metric CSVs, and visualization charts.

---

## 🛠️ Setup & Requirements

### 1. Python Dependencies
Ensure you are using Python 3.9+. Install the required dependencies:

```bash
pip install torch numpy pandas seaborn matplotlib tqdm boto3 botocore scikit-learn
```

### 2. AWS Authentication (gds-cli)
This framework utilizes Amazon Bedrock (Titan V2 embeddings and Claude 3.7 Sonnet). You must authenticate your AWS session via the Government Digital Service (GDS) CLI before running the scripts.

Ensure you have `gds-cli` installed and configured. Assume the appropriate AWS role containing Bedrock permissions:

```bash
gds aws <your-role-name> -e
```

> **Note:** The pipeline uses a token-bucket rate limiter via `botocore.config` to handle high-concurrency requests, but you must ensure your AWS session token remains active during execution.

---

## 🚀 Execution Guide

### Step 1: Generate the Datasets
Before running the main pipeline, you must generate the Phase I (Clean Baseline) and Phase II (Adversarial Paraphrased) datasets.

```bash
python data_generator.py
```
*   **What this does:** Generates `synthetic_corpus_phase1.json` (10k clean combinations of intents and contexts) and `synthetic_corpus_phase2.json` (10k LLM-paraphrased adversarial inputs).
*   **Output:** Files are saved directly to the `data/` directory.

### Step 2: Run the Main Experiment Pipeline
Execute the core Monte Carlo evaluation script.

```bash
python main.py
```
**What this does:**
*   Loads the generated corpora and generates Titan V2 embeddings concurrently (saving to `cache/`).
*   Evaluates the Zero-Shot LLM baseline.
*   Executes deterministic geometric routing (Handcrafted rules).
*   Optimizes system parameters via PyTorch BPTT across 5 variants.
*   Outputs deep JSON traces and aggregated metrics.

> **Time Warning:** Initial execution requires embedding 40,000 strings. Subsequent runs will use the local `cache/` and execute in seconds.

### Step 3: Run the Semantic Group Analysis
Once `main.py` finishes and the master traces are saved in `results/`, generate the vulnerability heatmaps.

```bash
python group_analysis.py
```
*   **What this does:** Computes the "Group Anchor" vector for each semantic intent and maps its geometric proximity to the destination vectors against its specific Destination Resolution Rate (DRR).

---

## Interpreting the Results

### Key Metrics
The framework outputs results based on two strict SLA metrics:

*   **DPR (Destination Protection Rate / Safety):** The percentage of queries that were safely handled. A query is "Safe" if it was correctly routed (SR), successfully trapped as Out-of-Domain (TN), or if the system safely abstained (SA) due to uncertainty. The baseline requirement is 100% Safety.
*   **DRR (Destination Resolution Rate / Yield):** The percentage of in-domain queries that were successfully matched and routed. A higher yield means fewer queries are handed off to the expensive System 2 LLM.

### Output Files (`results/` folder)

*   `*_results_chart.png`: A tri-pane Matplotlib bar chart showing Train, Test, and Full DPR/DRR comparisons for every evaluated architecture.
*   `*_tier_analysis.png`: A stacked bar chart breaking down how each model handled specific query difficulties (Direct ID vs. Ambiguous vs. Adversarial Traps).
*   `Phase_I_Group_Metrics.csv` & `Phase_II_Group_Metrics.csv`: Data linking geometric vector similarity to the precise yield of specific semantic groups.
*   `*_anchor_heatmap.png`: Seaborn heatmaps visually proving the "Yield Generalization Gap"—showing how queries with similarities below the minimum similarity floor trigger safe abstentions.
*   `*_traces_master.json`: Comprehensive interaction logs including the calculated entropy, maximum similarity scores, probability vectors, and exact turn-by-turn interactions for debugging.

---

## Approach Overview
This framework tests the hypothesis that a neuro-symbolic router (System 1) relying on fast vector-math (Cosine Similarity and Shannon Entropy) can guarantee 100% conversational safety while maximizing automated yield. By evaluating the system against both clean inputs (Phase I) and semantically degraded inputs (Phase II), the pipeline maps the exact mathematical boundaries where semantic drift forces the System 1 router to safely abstain and hand off control to a generative LLM (System 2).
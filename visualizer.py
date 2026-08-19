import os
import json
import pandas as pd
import matplotlib.pyplot as plt

class NSBRVisualizer:
    """
    Handles Matplotlib and Pandas-based charting for the NSBR framework's 
    stacked tier analysis.
    """

    @staticmethod
    def plot_triple_results(train_res, test_res, full_res, phase_title):
        """
        Generates a comparative bar chart of key metrics (DPR, DRR, SAR, FAR) 
        across Train, Test, and Full sets for each evaluated architecture.
        """
        print(f"Generating Triple Results Plot for: {phase_title}...")
        
        # Define the core metrics we want to plot
        metrics_to_plot = ["DPR", "DRR", "SAR", "FAR"]
        
        # Flatten the dictionaries into a single DataFrame
        records = []
        for split_name, data_dict in [("Train", train_res), ("Test", test_res), ("Full", full_res)]:
            for method, metrics in data_dict.items():
                # Clean up method names to fit nicely on the x-axis
                clean_method = method.replace('\n', ' ')
                
                record = {"Split": split_name, "Method": clean_method}
                for m in metrics_to_plot:
                    record[m] = metrics.get(m, 0.0)
                records.append(record)
                
        df = pd.DataFrame(records)
        
        if df.empty:
            print("Warning: No data provided to plot_triple_results.")
            return

        # Sort the methods by our established architecture groups
        df['sort_key'] = df['Method'].apply(lambda x: NSBRVisualizer._get_architecture_group(x)[0])
        df = df.sort_values(by=['sort_key', 'Method'])
        methods = df['Method'].unique()

        # Set up a 2x2 grid for the 4 metrics
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        axes = axes.flatten()
        fig.suptitle(f'{phase_title}: Core Metrics Across Data Splits', fontsize=18, fontweight='bold', y=1.02)

        # Standardized colors for Train, Test, Full
        split_colors = {"Train": "#aec7e8", "Test": "#1f77b4", "Full": "#2ca02c"}

        for i, metric in enumerate(metrics_to_plot):
            ax = axes[i]
            
            # Pivot data so Methods are on the X-axis and Splits are grouped bars
            pivot_df = df.pivot(index='Method', columns='Split', values=metric)
            
            # Reorder rows to match our architecture sorting
            pivot_df = pivot_df.reindex(methods)
            
            # Reorder columns to always be Train -> Test -> Full
            ordered_splits = [s for s in ["Train", "Test", "Full"] if s in pivot_df.columns]
            pivot_df = pivot_df[ordered_splits]
            
            # Plot
            pivot_df.plot(
                kind='bar', 
                color=[split_colors[s] for s in ordered_splits], 
                ax=ax, 
                width=0.7, 
                edgecolor='black', 
                linewidth=0.5
            )
            
            ax.set_title(f"{metric} (%)", fontsize=14, fontweight='bold', pad=10)
            ax.set_xlabel('')
            ax.set_ylabel('Percentage (%)')
            ax.set_ylim(0, 105)
            ax.set_xticklabels(pivot_df.index, rotation=45, ha='right', fontsize=10)
            
            # Add gridlines for easier reading
            ax.yaxis.grid(True, linestyle='--', alpha=0.7)
            ax.set_axisbelow(True)

            # Only keep the legend on the first subplot to reduce clutter
            if i == 0:
                ax.legend(title='Dataset Split', fontsize=10)
            else:
                if ax.get_legend() is not None:
                    ax.get_legend().remove()

        plt.tight_layout()
        
        # Save the figure
        os.makedirs("results", exist_ok=True)
        filename_safe_title = phase_title.replace(' ', '_').replace('(', '').replace(')', '').lower()
        output_filename = os.path.join("results", f"{filename_safe_title}_triple_results.png")
        
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"Successfully saved triple results plot to: {output_filename}")
        plt.close()


    @staticmethod
    def _categorize_tier(traj_id, tier_label):
        """Extracts the granular difficulty tier based on trajectory ID and base tier."""
        if 'ood_near' in traj_id:
            return 'Near OOD (Hard)'
        elif 'ood_far' in traj_id:
            return 'Far OOD (Baseline Noise)'
        elif tier_label == 'Ambiguous_ID':
            return 'Ambiguous ID (Medium)'
        elif tier_label == 'Direct_ID':
            return 'Direct ID (Easy)'
        return 'Unknown'

    @staticmethod
    def _map_outcome(outcome_code):
        """Maps the raw outcome codes to the analysis categories."""
        if outcome_code == 'SR':
            return 'Successfully Routed'
        elif outcome_code == 'MR':
            return 'Misrouted'
        elif outcome_code in ['SA', 'TN']:
            return 'Abstained / True Negative'
        return 'Other'
        
    @staticmethod
    def _get_architecture_group(method_name):
        """
        Assigns an integer sort key and group name based on the architecture type 
        to ensure they are clustered together on the x-axis.
        """
        name_lower = str(method_name).lower()
        if 'sonnet' in name_lower or 'llm' in name_lower or 'shot' in name_lower:
            return 1, 'Generative LLM'
        elif 'logistic' in name_lower or 'mlp' in name_lower:
            return 2, 'Classical ML'
        elif 'handcrafted' in name_lower or 'manual' in name_lower or 'heuristic' in name_lower:
            return 3, 'Heuristic'
        elif 'bptt' in name_lower or 'nsbr' in name_lower:
            return 4, 'BPTT'
        return 5, 'Other'

    @staticmethod
    def _process_tier_data(df):
        """Applies categorizations and calculates percentage distributions."""
        if df.empty:
            return df

        # Apply mapping logic for granular sub-tiers
        df['Granular_Tier'] = df.apply(
            lambda row: NSBRVisualizer._categorize_tier(
                row['trajectory_id'], 
                row['tier'] if 'tier' in df.columns else row.get('operational_tier', 'Unknown')
            ), 
            axis=1
        )
        df['Mapped_Outcome'] = df['outcome'].apply(NSBRVisualizer._map_outcome)

        # Group and calculate percentages
        grouped = df.groupby(['Phase', 'method', 'Granular_Tier', 'Mapped_Outcome']).size().reset_index(name='Count')
        totals = df.groupby(['Phase', 'method', 'Granular_Tier']).size().reset_index(name='Total')
        
        merged = pd.merge(grouped, totals, on=['Phase', 'method', 'Granular_Tier'])
        merged['Percentage'] = (merged['Count'] / merged['Total']) * 100
        
        return merged

    @staticmethod
    def plot_tier_analysis(file_path, phase_name):
        """Generates a 2x2 stacked bar chart for a specific phase from a JSON trace file."""
        if not os.path.exists(file_path):
            print(f"Warning: '{file_path}' not found. Cannot generate tier analysis.")
            return

        with open(file_path, 'r') as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)
        df['Phase'] = phase_name
        # Filter out pathological control baselines prior to generating bar charts
        df = df[~df['method'].str.contains('Unconstrained', case=False, na=False)]
        
        # Format the naming convention to match the LaTeX manuscript charts
        df['method'] = df['method'].str.replace('Manual', 'Handcrafted', regex=False)
        
        processed_df = NSBRVisualizer._process_tier_data(df)
        if processed_df.empty:
            return

        tier_order = ['Direct ID (Easy)', 'Ambiguous ID (Medium)', 'Near OOD (Hard)', 'Far OOD (Baseline Noise)']
        
        # Color mapping specific to the manuscript design
        colors = {
            'Successfully Routed': '#2ca02c',
            'Abstained / True Negative': '#1f77b4',
            'Misrouted': '#d62728'
        }

        # Change to 2x2 Grid, increase height for better spacing
        fig, axes = plt.subplots(2, 2, figsize=(22, 16), sharey=True)
        axes = axes.flatten() # Flatten 2x2 array for easy iteration
        
        fig.suptitle(f'{phase_name}: Routing Outcomes by Difficulty Tier across Architectures', 
                     fontsize=20, fontweight='bold', y=1.02)

        for i, tier in enumerate(tier_order):
            ax = axes[i]
            tier_data = processed_df[processed_df['Granular_Tier'] == tier]
            
            if tier_data.empty:
                ax.set_title(tier)
                continue
                
            pivot_df = tier_data.pivot(index='method', columns='Mapped_Outcome', values='Percentage').fillna(0)
            
            # Ensure all outcome categories map out safely, even if zeros
            for outcome in colors.keys():
                if outcome not in pivot_df.columns:
                    pivot_df[outcome] = 0.0
                    
            # Grouping and Sorting Logic (with spacers)

            # Assign sort keys
            pivot_df['sort_key'] = [NSBRVisualizer._get_architecture_group(x)[0] for x in pivot_df.index]

            if tier in ['Direct ID (Easy)', 'Ambiguous ID (Medium)']:
                # For ID queries, sort by highest Yield (Green)
                pivot_df = pivot_df.sort_values(by=['sort_key', 'Successfully Routed'], ascending=[True, False])
            else:
                # For OOD queries, sort by highest Safety/Abstention (Blue)
                pivot_df = pivot_df.sort_values(by=['sort_key', 'Abstained / True Negative'], ascending=[True, False])
            
            # Reconstruct index with empty spacer rows between groups
            new_records = []
            new_index = []
            current_group = None
            spacer_idx = 0
            
            for idx, row in pivot_df.iterrows():
                group_key = row['sort_key']
                if current_group is not None and group_key != current_group:
                    # Inject a visual spacer
                    spacer_name = f"__spacer_{spacer_idx}__"
                    new_index.append(spacer_name)
                    new_records.append({c: 0.0 for c in colors.keys()})
                    spacer_idx += 1
                
                new_index.append(idx)
                new_records.append({c: row[c] for c in colors.keys()})
                current_group = group_key
                
            spaced_df = pd.DataFrame(new_records, index=new_index)
            spaced_df = spaced_df[list(colors.keys())] # Drop sort_key

            # Plot the spaced dataframe
            spaced_df.plot(kind='bar', stacked=True, color=[colors[col] for col in spaced_df.columns], ax=ax, width=0.85)
            
            # Formatting
            ax.set_title(tier, fontsize=16, fontweight='bold', pad=15)
            ax.set_xlabel('')
            
            if i % 2 == 0:
                ax.set_ylabel('Percentage of Queries (%)', fontsize=12)
            
            ax.set_ylim(0, 105)
            
            # Format x-ticks (hide spacer names)
            cleaned_labels = [label if not label.startswith("__spacer") else "" for label in spaced_df.index]
            ax.set_xticklabels(cleaned_labels, rotation=45, ha='right', rotation_mode='anchor', fontsize=11)
            
            # Manage Legends
            if i == 1: # Top right plot
                ax.legend(title='Outcome', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, title_fontsize=14)
            else:
                if ax.get_legend() is not None:
                    ax.get_legend().remove()

        plt.tight_layout(w_pad=3.0, h_pad=4.0)
        
        # Ensure the results directory exists
        os.makedirs("results", exist_ok=True)
        output_filename = os.path.join("results", f"{phase_name.replace(' ', '_')}_tier_analysis_2x2.png")
        
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"Successfully generated and saved 2x2 tier analysis: {output_filename}")
        plt.close()

if __name__ == "__main__":
    # Define the paths to your JSON trace files
    phase1_traces = "results/phase1_grouped_test_traces_master.json"
    phase2_traces = "results/phase2_grouped_test_traces_master.json"
    
    # Generate the tier plots
    NSBRVisualizer.plot_tier_analysis(phase1_traces, "Phase I")
    NSBRVisualizer.plot_tier_analysis(phase2_traces, "Phase II")
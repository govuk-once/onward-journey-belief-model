import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class NSBRVisualizer:
    """
    Handles all Matplotlib and Pandas-based charting for the NSBR framework,
    including the triple-pane SLA results and the stacked tier analysis.
    """

    @staticmethod
    def plot_triple_results(train_results, test_results, full_results, phase_name):
        """Generates Tri-Pane Presentation Chart (Train/Test/Full) for QPR and DRR SLAs."""
        print(f"\n--- Generating Tri-Pane Presentation Chart for {phase_name} ---")
        labels = list(train_results.keys())
        x = np.arange(len(labels))
        width = 0.35

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(24, 18), sharex=True)

        def plot_pane(ax, data_dict, title):
            qpr_scores = [data_dict[l]["QPR"] for l in labels]
            drr_scores = [data_dict[l]["DRR"] for l in labels]
            
            rects1 = ax.bar(x - width/2, qpr_scores, width, label='QPR (Safety)', color='#2ecc71', edgecolor='black')
            rects2 = ax.bar(x + width/2, drr_scores, width, label='DRR (Yield)', color='#3498db', edgecolor='black')
            
            ax.set_ylabel('SLA (%)', fontsize=12, fontweight='bold')
            ax.set_title(title, fontsize=14, fontweight='bold', pad=12)
            ax.set_ylim(0, 115)
            ax.legend(loc='upper right', framealpha=0.9, fontsize=10)
            ax.axhline(100, color='#e74c3c', linestyle='--', linewidth=1.5, alpha=0.7)

            for rect in rects1 + rects2:
                height = rect.get_height()
                ax.annotate(f'{height:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 4), textcoords="offset points",
                            ha='center', va='bottom', fontsize=8, fontweight='bold')

        plot_pane(ax1, train_results, f'{phase_name} - TRAIN SET (80%)')
        plot_pane(ax2, test_results,  f'{phase_name} - TEST SET (20%)')
        plot_pane(ax3, full_results,  f'{phase_name} - FULL CORPUS (100%)')

        ax3.set_xticks(x)
        ax3.set_xticklabels(labels, rotation=45, ha="right", fontsize=10, fontweight='bold')

        fig.tight_layout()
        chart_path = os.path.join("results", f'{phase_name.lower().replace(" ", "_")}_results_chart.png')
        plt.savefig(chart_path, dpi=300)
        print(f"Chart successfully saved to: '{chart_path}'")
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
    def _process_tier_data(df):
        """Applies categorizations and calculates percentage distributions."""
        if df.empty:
            return df

        # Apply mapping logic for granular sub-tiers
        df['Granular_Tier'] = df.apply(lambda row: NSBRVisualizer._categorize_tier(row['trajectory_id'], row['tier']), axis=1)
        df['Mapped_Outcome'] = df['outcome'].apply(NSBRVisualizer._map_outcome)

        # Group and calculate percentages
        grouped = df.groupby(['Phase', 'method', 'Granular_Tier', 'Mapped_Outcome']).size().reset_index(name='Count')
        totals = df.groupby(['Phase', 'method', 'Granular_Tier']).size().reset_index(name='Total')
        
        merged = pd.merge(grouped, totals, on=['Phase', 'method', 'Granular_Tier'])
        merged['Percentage'] = (merged['Count'] / merged['Total']) * 100
        
        return merged

    @staticmethod
    def plot_tier_analysis(file_path, phase_name):
        """Generates a stacked bar chart for a specific phase from a JSON trace file."""
        if not os.path.exists(file_path):
            print(f"Warning: '{file_path}' not found. Cannot generate tier analysis.")
            return

        with open(file_path, 'r') as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)
        df['Phase'] = phase_name
        
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

        fig, axes = plt.subplots(1, 4, figsize=(20, 8), sharey=True)
        fig.suptitle(f'{phase_name}: Routing Outcomes by Difficulty Tier across Architectures', fontsize=18, fontweight='bold', y=1.05)

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
                    
            pivot_df = pivot_df[list(colors.keys())]

            pivot_df.plot(kind='bar', stacked=True, color=[colors[col] for col in pivot_df.columns], ax=ax, width=0.8)
            
            ax.set_title(tier, fontsize=14, fontweight='bold')
            ax.set_xlabel('')
            ax.set_ylabel('Percentage of Queries (%)' if i == 0 else '')
            ax.set_ylim(0, 100)
            ax.set_xticklabels(pivot_df.index, rotation=65, ha='right', rotation_mode='anchor', fontsize=10)
            
            if i == 3:
                ax.legend(title='Outcome', bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                ax.get_legend().remove()

        plt.tight_layout(w_pad=2.0)
        output_filename = os.path.join("results", f"{phase_name.replace(' ', '_')}_tier_analysis.png")
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"Successfully generated and saved tier analysis: {output_filename}")
        plt.close()
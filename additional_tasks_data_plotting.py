import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load your CSV files (make sure these filenames match your actual files)
df_micro = pd.read_csv("bo_only_micro.csv")
df_bench = pd.read_csv("bo_only_bench.csv")
df_pilot = pd.read_csv("bo_only_pilot.csv") 

plt.figure(figsize=(10, 6), dpi=300)

# ---------------------------------------------------------
# Helper function to compute both metrics
# ---------------------------------------------------------
def process_run_data(df):
    costs = df['cost_eur'].values
    yields = df['observed_Y'].values
    scales = df['scale'].values  # 0: micro, 1: bench, 2: pilot
    
    # We start at cumulative budget 0 with 0 yield to anchor the step plots cleanly
    cumulative_budget = np.insert(np.cumsum(costs), 0, 0.0)
    
    # Track overall best yield (including micro and bench)
    best_overall = np.insert(np.maximum.accumulate(yields), 0, 0.0)
    
    # Track validated pilot-only best yield
    best_pilot = []
    current_best_pilot = 0.0
    for y, scale in zip(yields, scales):
        if str(scale) in ['2', '2.0', 'pilot']:
            current_best_pilot = max(current_best_pilot, y)
        best_pilot.append(current_best_pilot)
        
    best_pilot = np.insert(np.array(best_pilot), 0, 0.0)
        
    return cumulative_budget, best_overall, best_pilot

# ---------------------------------------------------------
# Process and Plot
# ---------------------------------------------------------

# 1. Micro Run (Blue)
budget_m, overall_m, pilot_m = process_run_data(df_micro)
plt.plot(budget_m, overall_m, color='#1f77b4', linestyle='-', linewidth=2.5, 
         drawstyle='steps-post', label='Only Micro (Best Overall Yield)')
# Marker and vertical drop-off at the final validated step
plt.scatter(budget_m[-1], pilot_m[-1], color='#1f77b4', marker='o', s=100, zorder=5)
plt.vlines(budget_m[-1], ymin=pilot_m[-1], ymax=overall_m[-1], color='#1f77b4', 
           linestyle='--', alpha=0.7)

# 2. Bench Run (Orange)
budget_b, overall_b, pilot_b = process_run_data(df_bench)
plt.plot(budget_b, overall_b, color='#ff7f0e', linestyle='-', linewidth=2.5, 
         drawstyle='steps-post', label='Only Bench (Best Overall Yield)')
# Marker and vertical drop-off at the final validated step
plt.scatter(budget_b[-1], pilot_b[-1], color='#ff7f0e', marker='s', s=100, zorder=5)
plt.vlines(budget_b[-1], ymin=pilot_b[-1], ymax=overall_b[-1], color='#ff7f0e', 
           linestyle='--', alpha=0.7)

# 3. Multi-fidelity Pilot Run (Baseline - Green)
# Overall best and best pilot are identical here because we only evaluate Pilot scale
budget_p, overall_p, pilot_p = process_run_data(df_pilot)
plt.plot(budget_p, pilot_p, color='#2ca02c', linestyle='-', linewidth=2.5, 
         drawstyle='steps-post', label='Only Pilot (Active Multi-Fidelity Best)')

# ---------------------------------------------------------
# Formatting the Graph
# ---------------------------------------------------------
# Custom legend entries to clarify the solid vs. dashed lines
plt.plot([], [], color='gray', linestyle='-', label='Surrogate Optimization Progress (Any Scale)')
plt.plot([], [], color='gray', linestyle='--', label='Validation Discrepancy (Drop-off)')
plt.scatter([], [], color='gray', marker='o', label='Final Validated Pilot Performance')

plt.title("Optimization Efficiency: The Scale-Up Blindspot", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Cumulative Budget Spent (EUR)", fontsize=12)
plt.ylabel("Yield ($g/L$)", fontsize=12)
plt.xlim(0, 15000)
plt.ylim(bottom=0)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper left', fontsize=9, frameon=True)

plt.tight_layout()
plt.savefig("benchmark_comparison_dual_metric.png", dpi=300)
plt.show()
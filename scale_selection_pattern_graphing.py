import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("bo_campaign_history_run_3.csv")

scale_map = {0: 'Micro (s=0)', 1: 'Bench (s=1)', 2: 'Pilot (s=2)',
             0.0: 'Micro (s=0)', 1.0: 'Bench (s=1)', 2.0: 'Pilot (s=2)'}
df['scale_name'] = df['scale'].map(scale_map).fillna(df['scale'])

# Compute running best pilot Y
pilot_mask = df['scale'].astype(float) == 2.0
df['best_pilot_so_far'] = np.where(
    pilot_mask,
    df['observed_Y'].where(pilot_mask).cummax(),
    np.nan
)
df['best_pilot_so_far'] = df['best_pilot_so_far'].ffill()

iterations = np.arange(1, len(df) + 1)
colors = {0: '#1f77b4', 1: '#ff7f0e', 2: '#2ca02c'}
col_list = [colors.get(int(float(s)), 'gray') for s in df['scale']]

fig, ax1 = plt.subplots(figsize=(14, 5), dpi=300)

# --- Left axis: scale selection ---
ax1.scatter(iterations, df['scale_name'], c=col_list, s=100,
            edgecolors='black', alpha=0.8, zorder=3)
ax1.plot(iterations, df['scale_name'], color='gray',
         linestyle=':', alpha=0.4, zorder=2)
ax1.set_xlabel("BO Iteration Step", fontsize=10)
ax1.set_ylabel("Selected Scale", fontsize=10)
ax1.grid(True, linestyle='--', alpha=0.4)

# --- Right axis: Y values ---
ax2 = ax1.twinx()

# Plot observed Y for each scale separately so colors match
for scale_idx, color in colors.items():
    mask = df['scale'].astype(float) == float(scale_idx)
    ax2.scatter(iterations[mask], df['observed_Y'][mask],
                color=color, marker='x', s=60, alpha=0.5, zorder=2)

# Running best pilot Y as a bold line
ax2.plot(iterations, df['best_pilot_so_far'],
         color='#2ca02c', linewidth=2.0, linestyle='-',
         label='Best pilot Y so far', zorder=4)

# Mark the actual best pilot point
best_idx = df['best_pilot_so_far'].idxmax()
ax2.scatter(best_idx + 1, df['best_pilot_so_far'][best_idx],
            color='red', s=150, zorder=5,
            label=f"Best pilot Y = {df['best_pilot_so_far'][best_idx]:.2f} g/L")

ax2.set_ylabel("Observed Y (g/L)", fontsize=10)
ax2.legend(loc='upper left', fontsize=9)

# --- Legend for scale colors (left axis) ---
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4',
           markersize=8, label='Micro (s=0)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff7f0e',
           markersize=8, label='Bench (s=1)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ca02c',
           markersize=8, label='Pilot (s=2)'),
    Line2D([0], [0], marker='x', color='gray', markersize=8,
           label='Observed Y (x markers)'),
]
ax1.legend(handles=legend_elements, loc='upper right', fontsize=9)

plt.title("Scale-Selection History with Observed Y and Best Pilot Trajectory",
          fontsize=12, fontweight='bold', pad=10)
plt.tight_layout()
plt.savefig("scale_selection_pattern.png", dpi=300)
plt.show()
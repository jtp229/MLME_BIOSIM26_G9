import pandas as pd
import numpy as np

# 1. Load the collected data
raw_data_file = "noise_bias_data.csv"
try:
    df = pd.read_csv(raw_data_file)
except FileNotFoundError:
    print(f"Error: Could not find '{raw_data_file}'. Please run your probe script first.")
    exit(1)

print("=== MULTI-SCALE NOISE AND BIAS ANALYSIS ===\n")

# 2. Group by recipe and scale to compute mean and sample standard deviation (noise)
summary = df.groupby(['recipe_name', 'scale'])['Y'].agg(['mean', 'std', 'count']).reset_index()

# Separate pilot data to act as the reference truth for bias calculation
pilot_df = summary[summary['scale'] == 'pilot'][['recipe_name', 'mean']].rename(columns={'mean': 'pilot_mean'})

# Merge back to calculate bias: (scale_mean - pilot_mean)
analytics = pd.merge(summary, pilot_df, on='recipe_name')
analytics['bias'] = analytics['mean'] - analytics['pilot_mean']

# Clean up presentation names
analytics = analytics.rename(columns={'mean': 'Mean_Y', 'std': 'Noise_StdDev', 'count': 'Reps'})
analytics = analytics[['recipe_name', 'scale', 'Reps', 'Mean_Y', 'Noise_StdDev', 'bias']]

# Print the granular table to terminal
print(analytics.to_string(index=False))

# --- Save granular analysis to CSV ---
granular_csv = "scale_recipe_analytics.csv"
analytics.to_csv(granular_csv, index=False)
print(f"\n-> Saved granular analytics to: {granular_csv}")


print("\n=== AGGREGATED METRICS PER SCALE ===")
aggregated_rows = []

for scale in ['micro', 'bench', 'pilot']:
    scale_data = analytics[analytics['scale'] == scale]
    avg_noise = scale_data['Noise_StdDev'].mean()
    avg_bias = scale_data['bias'].mean()
    
    # Terminal display
    print(f"Scale: {scale.upper()}")
    print(f"  Avg Noise (StdDev): {avg_noise:.5f}" if not np.isnan(avg_noise) else "  Avg Noise (StdDev): NaN (Insufficient Reps)")
    print(f"  Avg Bias vs Pilot : {avg_bias:.5f}")
    
    # Store for CSV export
    aggregated_rows.append({
        "scale": scale,
        "avg_noise_stddev": avg_noise if not np.isnan(avg_noise) else "",
        "avg_bias_vs_pilot": avg_bias
    })

# --- Save aggregated metrics to CSV ---
aggregated_df = pd.DataFrame(aggregated_rows)
aggregated_csv = "scale_aggregated_metrics.csv"
aggregated_df.to_csv(aggregated_csv, index=False)
print(f"-> Saved aggregated metrics to: {aggregated_csv}")
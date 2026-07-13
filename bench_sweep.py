"""
bench_sweep.py
==============
Run 20 Latin Hypercube bench experiments to map the full 5D parameter space
before BO. This is the critical missing step — without it the GP only knows
about two recipes (T=30 and T=35.5) and will never find the high-yield region.

Cost: 20 x €500 = €10,000
Output: bench_sweep_results.csv
"""

import csv
import time
import numpy as np
from scipy.stats.qmc import LatinHypercube
import basic_client

client = basic_client.BioreactorClient(basic_client.BASE_URL)
client.login(basic_client.USER, basic_client.PASSWORD)

BOUNDS_LOW  = np.array([20.0, 3.0, 0.0, 0.0, 0.0])
BOUNDS_HIGH = np.array([60.0, 9.5, 2.0, 2.0, 2.0])
RECIPE_KEYS = ["T", "pH", "F1", "F2", "F3"]

# 20 LHS points — good coverage of the full 5D space
sampler = LatinHypercube(d=5, seed=42)
sweep_X = BOUNDS_LOW + sampler.random(20) * (BOUNDS_HIGH - BOUNDS_LOW)

print(f"Running {len(sweep_X)} bench experiments")
print(f"Expected cost: {len(sweep_X) * 500} EUR\n")

rows = []
for i, x in enumerate(sweep_X):
    recipe = {k: float(v) for k, v in zip(RECIPE_KEYS, x)}
    print(f"[{i+1:2d}/{len(sweep_X)}] "
          + "  ".join(f"{k}={v:.3f}" for k, v in recipe.items()),
          end="  ", flush=True)

    resp = client.run("bench", **recipe)
    y = float(resp["Y"])
    print(f"-> Y={y:.4f}  cost={resp['cost_eur']}€  total={resp['total_cost_eur']}€")

    rows.append({
        **recipe,
        "scale":          "bench",
        "Y":              y,
        "cost_eur":       resp["cost_eur"],
        "total_cost_eur": resp["total_cost_eur"],
        "run_id":         resp["id"],
    })
    time.sleep(0.3)

# Save
csv_filename = "bench_sweep_results.csv"
fieldnames = RECIPE_KEYS + ["scale", "Y", "cost_eur", "total_cost_eur", "run_id"]
with open(csv_filename, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

# Print summary to identify promising regions
ys = [r["Y"] for r in rows]
top5 = sorted(rows, key=lambda r: r["Y"], reverse=True)[:5]

print(f"\n{'='*60}")
print(f"Sweep complete — saved to {csv_filename}")
print(f"Y range: {min(ys):.4f} – {max(ys):.4f} g/L")
print(f"\nTop 5 recipes by bench Y:")
print(f"{'#':>2}  {'T':>6}  {'pH':>5}  {'F1':>5}  {'F2':>5}  {'F3':>5}  {'Y':>8}")
for i, r in enumerate(top5):
    print(f"{i+1:>2}  {r['T']:>6.2f}  {r['pH']:>5.3f}  "
          f"{r['F1']:>5.3f}  {r['F2']:>5.3f}  {r['F3']:>5.3f}  {r['Y']:>8.4f}")
print(f"{'='*60}")
print("Next step: run bo_multifidelity.py — it will load this file as seed data.")
"""
Latin Hypercube Sampling (LHS) sweep — Task 5 baseline + global exploration.

Strategy:
  1. Generate 20 well-spread bench runs across the full 5D space using LHS
  2. Run pilot on the top 2 bench recipes found
  3. Save everything to lhs_sweep_results.csv

Budget: 20 x 500 (bench) + 2 x 2000 (pilot) = 14,000 EUR
"""
import csv
import time
import numpy as np
from scipy.stats import qmc

import basic_client

client = basic_client.BioreactorClient(basic_client.BASE_URL)
client.login(basic_client.USER, basic_client.PASSWORD)

# --- Parameter bounds ---
PARAM_NAMES = ["T", "pH", "F1", "F2", "F3"]
LOWER = np.array([20.0, 3.0, 0.0, 0.0, 0.0])
UPPER = np.array([60.0, 9.5, 2.0, 2.0, 2.0])

N_BENCH  = 20   # LHS bench runs — broad exploration
N_PILOT  = 2    # pilot runs on top N_PILOT bench recipes

MOCK_MODE = False  # flip to False for real runs

# --- Mock function (same model as BO.py for consistency) ---
SCALE_NOISE = {0: 0.000350, 1: 0.003209, 2: 0.000006}
SCALE_COSTS = {0: 10.0, 1: 500.0, 2: 2000.0}
SCALE_MAPPING = {0: "micro", 1: "bench", 2: "pilot"}

def experiment_mock(recipe: np.ndarray, scale: int) -> tuple[float, float]:
    HOTSPOT = np.array([35.5, 6.5, 1.0, 1.0, 1.0])
    dist = np.linalg.norm(recipe - HOTSPOT)
    base = {0: 0.14, 1: 3.52, 2: 1.44}
    peak = {0: 0.18, 1: 5.42, 2: 2.36}
    t = max(0.0, 1.0 - dist / 10.0)
    y = base[scale] + t * (peak[scale] - base[scale])
    y += np.random.normal(0, np.sqrt(SCALE_NOISE[scale]))
    y = max(0.0, y)
    cost = SCALE_COSTS[scale]
    print(f"  MOCK -> scale={SCALE_MAPPING[scale]:<5s}  Y={y:.4f}  cost={cost}€")
    return y, cost

def experiment_real(recipe: np.ndarray, scale: int) -> tuple[float, float]:
    r = np.clip(recipe, LOWER, UPPER)
    recipe_dict = {k: float(v) for k, v in zip(PARAM_NAMES, r)}
    print(
        f"  API call -> scale={SCALE_MAPPING[scale]:<5s} | "
        + "  ".join(f"{k}={v:.3f}" for k, v in recipe_dict.items()),
        flush=True,
    )
    resp = client.run(SCALE_MAPPING[scale], **recipe_dict)
    y = float(resp["Y"])
    cost = SCALE_COSTS[scale]
    time.sleep(0.5)
    return y, cost

experiment = experiment_mock if MOCK_MODE else experiment_real

# ── Step 1: Generate LHS samples ──────────────────────────────────────────────
print(f"Generating {N_BENCH} LHS samples across 5D space...")
sampler = qmc.LatinHypercube(d=5, seed=42)
unit_samples = sampler.random(n=N_BENCH)                  # shape (N_BENCH, 5), values in [0,1]
recipes = qmc.scale(unit_samples, LOWER, UPPER)           # scale to actual bounds

print(f"Recipe space coverage:")
for i, name in enumerate(PARAM_NAMES):
    print(f"  {name}: [{recipes[:, i].min():.2f}, {recipes[:, i].max():.2f}]  "
          f"(full range: [{LOWER[i]:.1f}, {UPPER[i]:.1f}])")

# ── Step 2: Run bench on all LHS recipes ──────────────────────────────────────
print(f"\n--- Bench sweep ({N_BENCH} runs x 500 EUR = {N_BENCH*500} EUR) ---")
bench_results = []
total_spent = 0.0

for i, recipe in enumerate(recipes):
    print(f"\nBench run {i+1}/{N_BENCH} | total_spent={total_spent:.0f}€")
    y, cost = experiment(recipe, scale=1)
    total_spent += cost
    bench_results.append({
        "run":    i + 1,
        "scale":  "bench",
        "T":      float(recipe[0]),
        "pH":     float(recipe[1]),
        "F1":     float(recipe[2]),
        "F2":     float(recipe[3]),
        "F3":     float(recipe[4]),
        "Y":      y,
        "cost_eur":        cost,
        "cumulative_cost": total_spent,
        "source": "lhs_bench",
    })
    print(f"  Y={y:.4f}  cumulative={total_spent:.0f}€")

# ── Step 3: Rank bench results and pick top N_PILOT ───────────────────────────
bench_results.sort(key=lambda r: r["Y"], reverse=True)
print(f"\n--- Top {N_PILOT} bench recipes ---")
for i, r in enumerate(bench_results[:N_PILOT]):
    print(f"  #{i+1}: Y={r['Y']:.4f}  T={r['T']:.1f} pH={r['pH']:.2f} "
          f"F1={r['F1']:.2f} F2={r['F2']:.2f} F3={r['F3']:.2f}")

# ── Step 4: Run pilot on top recipes ─────────────────────────────────────────
print(f"\n--- Pilot runs ({N_PILOT} runs x 2000 EUR = {N_PILOT*2000} EUR) ---")
pilot_results = []

for i, top in enumerate(bench_results[:N_PILOT]):
    recipe = np.array([top["T"], top["pH"], top["F1"], top["F2"], top["F3"]])
    print(f"\nPilot run {i+1}/{N_PILOT} | total_spent={total_spent:.0f}€")
    y, cost = experiment(recipe, scale=2)
    total_spent += cost
    pilot_results.append({
        "run":    i + 1,
        "scale":  "pilot",
        "T":      float(recipe[0]),
        "pH":     float(recipe[1]),
        "F1":     float(recipe[2]),
        "F2":     float(recipe[3]),
        "F3":     float(recipe[4]),
        "Y":      y,
        "cost_eur":        cost,
        "cumulative_cost": total_spent,
        "source": "lhs_pilot",
    })
    print(f"  Y={y:.4f}  cumulative={total_spent:.0f}€")

# ── Step 5: Save all results ──────────────────────────────────────────────────
all_results = bench_results + pilot_results
# restore bench_results to original order for CSV
all_results_sorted = sorted(
    bench_results + pilot_results,
    key=lambda r: r["cumulative_cost"]
)

csv_filename = "lhs_sweep_results.csv"
fieldnames = ["run", "scale", "T", "pH", "F1", "F2", "F3",
              "Y", "cost_eur", "cumulative_cost", "source"]

with open(csv_filename, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_results_sorted)

# ── Summary ───────────────────────────────────────────────────────────────────
best_pilot_Y = max(r["Y"] for r in pilot_results) if pilot_results else 0.0
best_bench_Y = max(r["Y"] for r in bench_results)

print("\n" + "=" * 55)
print(f"LHS sweep complete — saved to {csv_filename}")
print(f"Total spent:       {total_spent:.0f} EUR")
print(f"Best bench Y:      {best_bench_Y:.4f} g/L")
print(f"Best pilot Y:      {best_pilot_Y:.4f} g/L  (Felix baseline: 14.0)")
print(f"\nTop 5 bench recipes found:")
print(f"  {'#':>3} {'T':>6} {'pH':>5} {'F1':>5} {'F2':>5} {'F3':>5} {'Y':>8}")
for i, r in enumerate(bench_results[:5]):
    print(f"  {i+1:>3} {r['T']:>6.1f} {r['pH']:>5.2f} {r['F1']:>5.2f} "
          f"{r['F2']:>5.2f} {r['F3']:>5.2f} {r['Y']:>8.4f}")
print("=" * 55)
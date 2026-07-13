"""
bo_multifidelity.py  —  Multi-Fidelity Bayesian Optimisation
=============================================================
Workflow:
  1. Run bench_sweep.py first  (EUR 10,000 — maps the full 5D space)
  2. Run this file              (EUR 5,000 remaining — BO refinement + pilot runs)

Bug fixes in this version:
  - PILOT_RESERVE ensures budget is always kept for pilot runs
  - Pilot runs are forced every N_BENCH_BEFORE_PILOT bench steps
  - best_f always uses pilot Y when available, never bench Y as fallback
    (bench Y ~2.37x higher than pilot — using it as best_f makes pilot EI tiny)
  - Login guarded behind MOCK_MODE so mock runs never hit the API
"""

import csv
import os
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

import basic_client

# ---------------------------------------------------------------------------
# MOCK_MODE must be set before client login
# ---------------------------------------------------------------------------
MOCK_MODE = False   # <- flip to False for real run

if not MOCK_MODE:
    client = basic_client.BioreactorClient(basic_client.BASE_URL)
    client.login(basic_client.USER, basic_client.PASSWORD)
else:
    client = None   # never touches API in mock mode

# ---------------------------------------------------------------------------
# Parameter space
# ---------------------------------------------------------------------------
RECIPE_MIN  = np.array([20.0,  3.0, 0.0, 0.0, 0.0])
RECIPE_MAX  = np.array([60.0,  9.5, 2.0, 2.0, 2.0])
RECIPE_KEYS = ["T", "pH", "F1", "F2", "F3"]

<<<<<<< HEAD
RECIPE_MIN = np.array([33.0,  7.0, 0.0, 0.0, 0.0])
RECIPE_MAX = np.array([38.0,  9.5, 2.0, 2.0, 2.0])
 
SCALE_MAPPING = {0: "micro", 1: "bench", 2: "pilot"}

# Set to a float to seek a target yield; None = maximize Y.
TARGET_PILOT_Y: float | None = None
 
# Confirmed from CSV: micro=10, bench=500, pilot=2000
SCALE_COSTS = {0: 10.0, 1: 500.0, 2: 2000.0}
 
# Noise variances (sigma^2) — pooled from empirical repeat data
# micro:  pooled var of baseline (0.000405) and hotspot (0.000196) repeats
# bench:  conservative estimate — only 2 hotspot reps available
# pilot:  assumed lower than bench per project spec
=======
SCALE_MAPPING = {0: "micro", 1: "bench", 2: "pilot"}
SCALE_COSTS   = {0: 10.0,    1: 500.0,   2: 2000.0}

>>>>>>> d60b506 (ok)
SCALE_NOISE = {
    0: 0.00050361,   # micro  — pooled empirical (sigma^2)
    1: 0.00093262,   # bench  — pooled empirical (sigma^2)
    2: 0.00002018,   # pilot  — pooled empirical (sigma^2)
}

B = np.array([
    [1.00, 0.10, 0.05],
    [0.10, 1.00, 0.90],
    [0.05, 0.90, 1.00],
])

# ---------------------------------------------------------------------------
# Budget strategy
# ---------------------------------------------------------------------------
BUDGET_LIMIT         = 5000.0   # total budget for this BO run
PILOT_RESERVE        = 4000.0   # always keep this much for pilot runs
                                 # = 2 pilot runs (confirmation + 1 mid-campaign)
N_BENCH_BEFORE_PILOT = 5        # force a pilot run every N bench/micro steps

# ---------------------------------------------------------------------------
# Multi-fidelity kernel
# ---------------------------------------------------------------------------
class MultiFidelityKernel(RBF):
    def __call__(self, X, Y=None, eval_gradient=False):
        X_recipe = X[:, :5]
        Y_recipe = Y[:, :5] if Y is not None else None

        if eval_gradient:
            rbf_matrix, rbf_gradient = super().__call__(
                X_recipe, Y_recipe, eval_gradient=True
            )
        else:
            rbf_matrix = super().__call__(
                X_recipe, Y_recipe, eval_gradient=False
            )

        n_X = X.shape[0]
        n_Y = Y.shape[0] if Y is not None else n_X
        b_matrix = np.zeros((n_X, n_Y))
        for i in range(n_X):
            for j in range(n_Y):
                si = int(X[i, 5])
                sj = int(Y[j, 5]) if Y is not None else int(X[j, 5])
                b_matrix[i, j] = B[si, sj]

        if eval_gradient:
            return rbf_matrix * b_matrix, rbf_gradient * b_matrix[:, :, np.newaxis]
        return rbf_matrix * b_matrix

# ---------------------------------------------------------------------------
# Acquisition function
# ---------------------------------------------------------------------------
def expected_improvement(mean: float, std: float, best_f: float) -> float:
    if std <= 1e-6:
        return 0.0
    z  = (mean - best_f) / std
    ei = (mean - best_f) * norm.cdf(z) + std * norm.pdf(z)
    return float(max(0.0, ei))

<<<<<<< HEAD
def expected_target_utility(mean: float, std: float, target: float) -> float:
    """
    Utility for *hitting a target* rather than maximizing Y.
    We maximize negative expected squared error:
        U = -E[(Y-target)^2] = -((mean-target)^2 + std^2)
    """
    return -(((mean - target) ** 2) + (std ** 2))


=======
# ---------------------------------------------------------------------------
# Next experiment selection
# ---------------------------------------------------------------------------
>>>>>>> d60b506 (ok)
def select_next_experiment(
    gp_model:     GaussianProcessRegressor,
    X_train:      np.ndarray,
    Y_train:      np.ndarray,
    force_pilot:  bool = False,
    allow_scales: list[int] = [0, 1, 2],
    n_candidates: int = 3000,
) -> tuple[np.ndarray, int]:
    """
    Bug fix: best_f is always derived from pilot observations only.
    Bench Y is ~2.37x higher than pilot — using it as best_f makes
    pilot EI negligibly small and the algorithm never selects pilot.

    If no pilot observations exist yet, best_f is set conservatively to 0
    so pilot runs look attractive from the start.
    """
    pilot_mask = X_train[:, 5] == 2
<<<<<<< HEAD
    if TARGET_PILOT_Y is None:
        best_f = float(np.max(Y_train))
    else:
        best_f = float("nan")
 
=======
    if np.any(pilot_mask):
        best_f = float(np.max(Y_train[pilot_mask]))   # pilot Y only
    else:
        best_f = 0.0   # conservative — makes pilot EI large, encouraging pilot selection

>>>>>>> d60b506 (ok)
    best_utility = -np.inf
    best_recipe  = None
    best_scale   = None

    random_recipes = np.random.uniform(RECIPE_MIN, RECIPE_MAX, size=(n_candidates, 5))

    # If forced pilot, only evaluate scale=2
    scales_to_check = [2] if force_pilot else allow_scales

    for scale_idx in scales_to_check:
        X_cand = np.column_stack([
            random_recipes,
            np.full(n_candidates, scale_idx, dtype=float),
        ])
        means, stds = gp_model.predict(X_cand, return_std=True)

        for recipe, mean, std in zip(random_recipes, means, stds):
<<<<<<< HEAD
            if TARGET_PILOT_Y is None:
                acq = expected_improvement(float(mean), float(std), best_f)
            else:
                acq = expected_target_utility(float(mean), float(std), TARGET_PILOT_Y)
            utility = acq / SCALE_COSTS[scale_idx]
 
=======
            ei      = expected_improvement(float(mean), float(std), best_f)
            utility = ei / SCALE_COSTS[scale_idx]
>>>>>>> d60b506 (ok)
            if utility > best_utility:
                best_utility = utility
                best_recipe  = recipe.copy()
                best_scale   = scale_idx

    return best_recipe, best_scale

# ---------------------------------------------------------------------------
# Live API experiment (defined first so mock override below works)
# ---------------------------------------------------------------------------
def experiment(recipe: np.ndarray, scale: int) -> tuple[float, float]:
    if client is None:
        raise RuntimeError("client is None — set MOCK_MODE = False")
    r = np.clip(recipe, RECIPE_MIN, RECIPE_MAX)
    recipe_dict = {k: float(v) for k, v in zip(RECIPE_KEYS, r)}
    print(
        f"  API -> scale={SCALE_MAPPING[scale]:<5s} | "
        + "  ".join(f"{k}={v:.3f}" for k, v in recipe_dict.items()),
        flush=True,
    )
    resp = client.run(SCALE_MAPPING[scale], **recipe_dict)
    y    = float(resp["Y"])
    cost = float(resp.get("cost_eur", SCALE_COSTS[scale]))
    basic_client.time.sleep(2.0)
    return y, cost

<<<<<<< HEAD


MOCK_MODE = False   # flip to False for real runs

def experiment_mock(recipe: np.ndarray, scale: int) -> tuple[float, float]:
    HOTSPOT = np.array([35.2, 8.05, 1.85, 1.03, 1.8])
=======
# ---------------------------------------------------------------------------
# Mock experiment
# ---------------------------------------------------------------------------
def experiment_mock(recipe: np.ndarray, scale: int) -> tuple[float, float]:
    """
    Mock anchored to your empirical data.
    bench/pilot ratio ~2.37x matches real observations.
    """
    HOTSPOT = np.array([35.5, 6.5, 1.0, 1.0, 1.0])
>>>>>>> d60b506 (ok)
    dist    = np.linalg.norm(recipe - HOTSPOT)
    base  = {0: 0.152, 1: 3.550, 2: 1.436}
    peak  = {0: 0.181, 1: 5.401, 2: 2.370}
    t     = max(0.0, 1.0 - dist / 15.0)
    y     = base[scale] + t * (peak[scale] - base[scale])
    y    += np.random.normal(0.0, np.sqrt(SCALE_NOISE[scale]))
    y     = max(0.0, y)
    cost  = SCALE_COSTS[scale]
    print(f"  MOCK -> scale={SCALE_MAPPING[scale]:<5s}  Y={y:.4f}  cost={cost}EUR")
    return y, cost

if MOCK_MODE:
    experiment = experiment_mock

# ---------------------------------------------------------------------------
# Seed data loader
# ---------------------------------------------------------------------------
def load_seed_data(sweep_csv: str = "bench_sweep_results.csv") -> list[dict]:
    if os.path.exists(sweep_csv):
        print(f"Loading bench sweep seed from {sweep_csv}")
        rows = []
        with open(sweep_csv) as f:
            for row in csv.DictReader(f):
                rows.append({
                    "recipe": [float(row[k]) for k in RECIPE_KEYS],
                    "scale":  1,
                    "Y":      float(row["Y"]),
                })
        print(f"  Loaded {len(rows)} bench observations")
        return rows

<<<<<<< HEAD
if __name__ == "__main__":
    BUDGET_LIMIT = 15000.0
 
   
    historical_seed = [
        # recipe                                   scale  Y
        ([30.0, 6.5, 1.0, 1.0, 1.0],               0,   0.12332),
        ([30.0, 6.5, 1.0, 1.0, 1.0],               0,   0.13581),
        ([30.0, 6.5, 1.0, 1.0, 1.0],               0,   0.16157),
        ([30.0, 6.5, 1.0, 1.0, 1.0],               0,   0.18744),
        ([30.0, 6.5, 1.0, 1.0, 1.0],               0,   0.13659),
        ([30.0, 6.5, 1.0, 1.0, 1.0],               0,   0.15115),
        ([30.0, 6.5, 1.0, 1.0, 1.0],               0,   0.13040),
        ([30.0, 6.5, 1.0, 1.0, 1.0],               0,   0.13712),
        ([30.0, 6.5, 1.0, 1.0, 1.0],               0,   0.12722),
        ([30.0, 6.5, 1.0, 1.0, 1.0],               0,   0.12164),
        ([30.0, 6.5, 1.0, 1.0, 1.0],               1,   3.53953),  # bench
        ([30.0, 6.5, 1.0, 1.0, 1.0],               1,   3.49412),  # bench
        ([30.0, 6.5, 1.0, 1.0, 1.0],               2,   1.43674),  # pilot ← most valuable
        ([35.5, 6.5, 1.0, 1.0, 1.0],               0,   0.18945),
        ([35.5, 6.5, 1.0, 1.0, 1.0],               0,   0.19557),
        ([35.5, 6.5, 1.0, 1.0, 1.0],               0,   0.16371),
        ([35.5, 6.5, 1.0, 1.0, 1.0],               0,   0.18316),
        ([35.5, 6.5, 1.0, 1.0, 1.0],               0,   0.16034),
        ([35.5, 6.5, 1.0, 1.0, 1.0],               0,   0.16540),
        ([35.5, 6.5, 1.0, 1.0, 1.0],               0,   0.16537),
        ([35.5, 6.5, 1.0, 1.0, 1.0],               0,   0.18353),
        ([35.5, 6.5, 1.0, 1.0, 1.0],               0,   0.16238),
        ([35.5, 6.5, 1.0, 1.0, 1.0],               0,   0.20264),
        ([35.5, 6.5, 1.0, 1.0, 1.0],               1,   5.33658),  # bench
        ([35.5, 6.5, 1.0, 1.0, 1.0],               1,   5.50103),  # bench
        ([35.5, 6.5, 1.0, 1.0, 1.0],               2,   2.35823),  # pilot
        # High-yield region discovered in prior BO campaigns (pH ~8, T ~35)
        ([35.595, 7.911, 1.922, 0.109, 1.651],     2,  21.18010),
        ([35.703, 7.997, 1.918, 0.941, 1.951],     2,  18.34820),
        ([35.684, 8.349, 1.848, 1.249, 1.833],     2,  14.48520),
        ([35.769, 8.098, 1.912, 0.639, 1.970],     1,  31.99850),
        ([35.979, 8.284, 1.841, 0.284, 1.305],     1,  30.29950),
        ([34.535, 8.071, 1.859, 0.759, 0.215],     2,   7.22931),
    ]
    # ─────────────────────────────────────────────────────────────────────
=======
    print("WARNING: bench_sweep_results.csv not found — using historical fallback seed.")
    print("         Run bench_sweep.py first for best results.\n")
    return [
        # baseline (T=30) — micro x10, bench x5, pilot x3
        {"recipe": [30.0, 6.5, 1.0, 1.0, 1.0], "scale": 0, "Y": 0.16052},
        {"recipe": [30.0, 6.5, 1.0, 1.0, 1.0], "scale": 0, "Y": 0.12923},
        {"recipe": [30.0, 6.5, 1.0, 1.0, 1.0], "scale": 0, "Y": 0.11726},
        {"recipe": [30.0, 6.5, 1.0, 1.0, 1.0], "scale": 0, "Y": 0.15758},
        {"recipe": [30.0, 6.5, 1.0, 1.0, 1.0], "scale": 0, "Y": 0.13262},
        {"recipe": [30.0, 6.5, 1.0, 1.0, 1.0], "scale": 0, "Y": 0.19308},
        {"recipe": [30.0, 6.5, 1.0, 1.0, 1.0], "scale": 0, "Y": 0.13349},
        {"recipe": [30.0, 6.5, 1.0, 1.0, 1.0], "scale": 0, "Y": 0.17679},
        {"recipe": [30.0, 6.5, 1.0, 1.0, 1.0], "scale": 0, "Y": 0.15127},
        {"recipe": [30.0, 6.5, 1.0, 1.0, 1.0], "scale": 0, "Y": 0.16371},
        {"recipe": [30.0, 6.5, 1.0, 1.0, 1.0], "scale": 1, "Y": 3.56226},
        {"recipe": [30.0, 6.5, 1.0, 1.0, 1.0], "scale": 1, "Y": 3.52130},
        {"recipe": [30.0, 6.5, 1.0, 1.0, 1.0], "scale": 1, "Y": 3.56868},
        {"recipe": [30.0, 6.5, 1.0, 1.0, 1.0], "scale": 1, "Y": 3.56982},
        {"recipe": [30.0, 6.5, 1.0, 1.0, 1.0], "scale": 1, "Y": 3.52744},
        {"recipe": [30.0, 6.5, 1.0, 1.0, 1.0], "scale": 2, "Y": 1.43426},
        {"recipe": [30.0, 6.5, 1.0, 1.0, 1.0], "scale": 2, "Y": 1.43871},
        {"recipe": [30.0, 6.5, 1.0, 1.0, 1.0], "scale": 2, "Y": 1.43382},
        # hotspot (T=35.5) — micro x10, bench x5, pilot x3
        {"recipe": [35.5, 6.5, 1.0, 1.0, 1.0], "scale": 0, "Y": 0.16548},
        {"recipe": [35.5, 6.5, 1.0, 1.0, 1.0], "scale": 0, "Y": 0.20186},
        {"recipe": [35.5, 6.5, 1.0, 1.0, 1.0], "scale": 0, "Y": 0.16756},
        {"recipe": [35.5, 6.5, 1.0, 1.0, 1.0], "scale": 0, "Y": 0.17153},
        {"recipe": [35.5, 6.5, 1.0, 1.0, 1.0], "scale": 0, "Y": 0.22004},
        {"recipe": [35.5, 6.5, 1.0, 1.0, 1.0], "scale": 0, "Y": 0.17717},
        {"recipe": [35.5, 6.5, 1.0, 1.0, 1.0], "scale": 0, "Y": 0.15566},
        {"recipe": [35.5, 6.5, 1.0, 1.0, 1.0], "scale": 0, "Y": 0.16646},
        {"recipe": [35.5, 6.5, 1.0, 1.0, 1.0], "scale": 0, "Y": 0.20821},
        {"recipe": [35.5, 6.5, 1.0, 1.0, 1.0], "scale": 0, "Y": 0.17944},
        {"recipe": [35.5, 6.5, 1.0, 1.0, 1.0], "scale": 1, "Y": 5.41808},
        {"recipe": [35.5, 6.5, 1.0, 1.0, 1.0], "scale": 1, "Y": 5.41184},
        {"recipe": [35.5, 6.5, 1.0, 1.0, 1.0], "scale": 1, "Y": 5.44598},
        {"recipe": [35.5, 6.5, 1.0, 1.0, 1.0], "scale": 1, "Y": 5.36192},
        {"recipe": [35.5, 6.5, 1.0, 1.0, 1.0], "scale": 1, "Y": 5.36526},
        {"recipe": [35.5, 6.5, 1.0, 1.0, 1.0], "scale": 2, "Y": 2.36875},
        {"recipe": [35.5, 6.5, 1.0, 1.0, 1.0], "scale": 2, "Y": 2.36522},
        {"recipe": [35.5, 6.5, 1.0, 1.0, 1.0], "scale": 2, "Y": 2.37646},
    ]

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    seed_data = load_seed_data("bench_sweep_results.csv")
>>>>>>> d60b506 (ok)

    history: list[dict] = []
    X_list,  Y_list     = [], []

    for entry in seed_data:
        x_row = np.append(entry["recipe"], entry["scale"])
        X_list.append(x_row)
        Y_list.append(entry["Y"])
        history.append({
            "T":             entry["recipe"][0],
            "pH":            entry["recipe"][1],
            "F1":            entry["recipe"][2],
            "F2":            entry["recipe"][3],
            "F3":            entry["recipe"][4],
            "scale":         entry["scale"],
            "observed_Y":    entry["Y"],
            "cost_eur":      0.0,
            "cumulative_cost": 0.0,
            "source":        "seed",
        })

    X_train = np.array(X_list)
    Y_train = np.array(Y_list)
    total_spent = 0.0

    scale_counts = {SCALE_MAPPING[s]: int(np.sum(X_train[:, 5] == s)) for s in [0, 1, 2]}
    print(f"Seeded GP with {len(X_list)} observations: {scale_counts}")

    bench_mask = X_train[:, 5] == 1
    if np.any(bench_mask):
        best_bench_idx = int(np.argmax(Y_train[bench_mask]))
        best_bench_x   = X_train[bench_mask][best_bench_idx]
        print(f"Best bench Y in seed: {np.max(Y_train[bench_mask]):.4f} g/L")
        print(f"  at T={best_bench_x[0]:.2f}  pH={best_bench_x[1]:.3f}  "
              f"F1={best_bench_x[2]:.3f}  F2={best_bench_x[3]:.3f}  F3={best_bench_x[4]:.3f}")

    mf_kernel   = MultiFidelityKernel(length_scale=5.0)
    alpha_noise = np.array([SCALE_NOISE[int(s)] for s in X_train[:, 5]])
    gp = GaussianProcessRegressor(
        kernel=mf_kernel,
        alpha=alpha_noise,
        n_restarts_optimizer=5,
        normalize_y=True,
    )

    # Budget split:
    #   PILOT_RESERVE   = reserved for pilot runs (never spent on micro/bench)
    #   exploration_cap = what micro/bench can spend freely
    exploration_cap = BUDGET_LIMIT - PILOT_RESERVE
    steps_since_pilot = 0
    best_pilot        = 0.0

    print(f"\n--- Launching BO loop ---")
    print(f"    Total budget:      EUR {BUDGET_LIMIT:.0f}")
    print(f"    Exploration cap:   EUR {exploration_cap:.0f}  (micro + bench)")
    print(f"    Pilot reserve:     EUR {PILOT_RESERVE:.0f}  (~{int(PILOT_RESERVE/2000)} pilot runs)")
    print(f"    Force pilot every: {N_BENCH_BEFORE_PILOT} non-pilot steps\n")

    iteration = 0
    while total_spent < BUDGET_LIMIT:
        remaining  = BUDGET_LIMIT - total_spent
        pilot_mask = X_train[:, 5] == 2

        # ── Decide which scales are affordable and allowed ────────────────
        exploration_spent = sum(
            h["cost_eur"] for h in history if h["scale"] in [0, 1]
        )
        exploration_remaining = exploration_cap - exploration_spent

        # Force pilot if:
        #   a) we've done N_BENCH_BEFORE_PILOT steps without one, OR
        #   b) exploration budget is exhausted but pilot reserve remains
        force_pilot = (
            steps_since_pilot >= N_BENCH_BEFORE_PILOT
            or exploration_remaining <= 0
        )

        if force_pilot and remaining >= SCALE_COSTS[2]:
            allow_scales = [2]
        elif force_pilot and remaining < SCALE_COSTS[2]:
            print("Wanted pilot but can't afford it — stopping early.")
            break
        else:
            # Only allow micro/bench while exploration budget remains
            allow_scales = [s for s in [0, 1] if SCALE_COSTS[s] <= exploration_remaining]
            if not allow_scales:
                # Exploration exhausted — switch to pilot only
                allow_scales = [2] if remaining >= SCALE_COSTS[2] else []
            if not allow_scales:
                break

        print(f"\n[iter {iteration:3d}]  spent={total_spent:.0f}  remaining={remaining:.0f}  "
              f"steps_since_pilot={steps_since_pilot}  force_pilot={force_pilot}  "
              f"best_pilot={best_pilot:.4f}", flush=True)

        gp.alpha = np.array([SCALE_NOISE[int(s)] for s in X_train[:, 5]])
        gp.fit(X_train, Y_train)

        next_recipe, next_scale = select_next_experiment(
            gp, X_train, Y_train,
            force_pilot=force_pilot,
            allow_scales=allow_scales,
        )

        y, cost = experiment(next_recipe, next_scale)
        total_spent += cost

        X_train = np.vstack([X_train, np.append(next_recipe, next_scale)])
        Y_train = np.append(Y_train, y)
<<<<<<< HEAD
 
        pilot_mask = X_train[:, 5] == 2
        best_overall = float(np.max(Y_train))
        if np.any(pilot_mask):
            if TARGET_PILOT_Y is None:
                best_pilot = float(np.max(Y_train[pilot_mask]))
            else:
                pilot_vals = Y_train[pilot_mask]
                best_pilot = float(pilot_vals[np.argmin(np.abs(pilot_vals - TARGET_PILOT_Y))])
        else:
            best_pilot = 0.0
 
=======

        if next_scale == 2:
            best_pilot        = float(np.max(Y_train[X_train[:, 5] == 2]))
            steps_since_pilot = 0
        else:
            steps_since_pilot += 1
>>>>>>> d60b506 (ok)

        history.append({
            "T":             float(next_recipe[0]),
            "pH":            float(next_recipe[1]),
            "F1":            float(next_recipe[2]),
            "F2":            float(next_recipe[3]),
            "F3":            float(next_recipe[4]),
            "scale":         next_scale,
            "observed_Y":    y,
            "cost_eur":      cost,
            "cumulative_cost": total_spent,
            "source":        "bo_loop",
        })
<<<<<<< HEAD
 
        print(
            f"  -> Y={y:.4f}  cost={cost:.0f}€  "
            f"total={total_spent:.0f}€  best={best_overall:.4f}  best_pilot={best_pilot:.4f}",
            flush=True,
        )
 
    # ── Save results ──────────────────────────────────────────────────────
=======

        print(f"  -> Y={y:.4f}  cost={cost:.0f}EUR  "
              f"total={total_spent:.0f}EUR  best_pilot={best_pilot:.4f}",
              flush=True)

        iteration += 1

    # ── Phase 3: final pilot confirmation of the best known recipe ────────
    print("\n--- Phase 3: final pilot confirmation ---")
    remaining = BUDGET_LIMIT - total_spent
    if remaining >= SCALE_COSTS[2]:
        best_idx    = int(np.argmax(Y_train))
        best_recipe = X_train[best_idx, :5]
        print(f"Best recipe overall: "
              f"T={best_recipe[0]:.2f}  pH={best_recipe[1]:.3f}  "
              f"F1={best_recipe[2]:.3f}  F2={best_recipe[3]:.3f}  "
              f"F3={best_recipe[4]:.3f}")
        y, cost = experiment(best_recipe, 2)
        total_spent += cost
        X_train = np.vstack([X_train, np.append(best_recipe, 2)])
        Y_train = np.append(Y_train, y)
        best_pilot = float(np.max(Y_train[X_train[:, 5] == 2]))
        history.append({
            "T": float(best_recipe[0]), "pH": float(best_recipe[1]),
            "F1": float(best_recipe[2]), "F2": float(best_recipe[3]),
            "F3": float(best_recipe[4]),
            "scale": 2, "observed_Y": y,
            "cost_eur": cost, "cumulative_cost": total_spent,
            "source": "confirmation",
        })
        print(f"  Pilot confirmation Y={y:.4f}  best_pilot={best_pilot:.4f}")
    else:
        print(f"  Insufficient budget (need EUR 2000, have EUR {remaining:.0f})")

    # ── Save ──────────────────────────────────────────────────────────────
>>>>>>> d60b506 (ok)
    csv_filename = "bo_campaign_history.csv"
    fieldnames   = ["T", "pH", "F1", "F2", "F3", "scale",
                    "observed_Y", "cost_eur", "cumulative_cost", "source"]
    with open(csv_filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
<<<<<<< HEAD
        writer.writerows(history)   # Bug 3 fix: all rows present, no zip mismatch
 
    # ── Final summary ─────────────────────────────────────────────────────
    pilot_mask = X_train[:, 5] == 2
    best_overall_Y = float(np.max(Y_train))
    if np.any(pilot_mask):
        if TARGET_PILOT_Y is None:
            best_pilot_Y = float(np.max(Y_train[pilot_mask]))
        else:
            pilot_vals = Y_train[pilot_mask]
            best_pilot_Y = float(pilot_vals[np.argmin(np.abs(pilot_vals - TARGET_PILOT_Y))])
    else:
        best_pilot_Y = 0.0
    n_by_scale  = {SCALE_MAPPING[s]: int(np.sum(X_train[:, 5] == s)) for s in [0, 1, 2]}

    print("\n" + "=" * 55)
    print(f"Campaign complete — results saved to {csv_filename}")
    print(f"Runs by scale:   {n_by_scale}")
    print(f"Total cost:      {total_spent:.0f} EUR")
    if TARGET_PILOT_Y is None:
        print(f"Best overall Y:  {best_overall_Y:.4f} g/L")
        print(f"Best pilot Y:    {best_pilot_Y:.4f} g/L")
    else:
        print(
            f"Best pilot Y:    {best_pilot_Y:.4f} g/L   "
            f"(target: {TARGET_PILOT_Y:.1f}, abs err: {abs(best_pilot_Y - TARGET_PILOT_Y):.4f})"
        )
=======
        writer.writerows(history)

    pilot_mask  = X_train[:, 5] == 2
    max_pilot_Y = float(np.max(Y_train[pilot_mask])) if np.any(pilot_mask) else 0.0
    n_by_scale  = {SCALE_MAPPING[s]: int(np.sum(X_train[:, 5] == s)) for s in [0, 1, 2]}

    print("\n" + "=" * 55)
    print(f"Results saved to {csv_filename}")
    print(f"Runs by scale:  {n_by_scale}")
    print(f"Total cost:     {total_spent:.0f} EUR")
    print(f"Best pilot Y:   {max_pilot_Y:.4f} g/L   (Felix baseline: 14.0)")
>>>>>>> d60b506 (ok)
    print("=" * 55)
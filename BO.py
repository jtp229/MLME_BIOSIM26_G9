
import csv
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
 
import basic_client

client = basic_client.BioreactorClient(basic_client.BASE_URL)
client.login(basic_client.USER, basic_client.PASSWORD)
 

RECIPE_MIN = np.array([20.0,  3.0, 0.0, 0.0, 0.0])
RECIPE_MAX = np.array([60.0,  9.5, 2.0, 2.0, 2.0])
 
SCALE_MAPPING = {0: "micro", 1: "bench", 2: "pilot"}
 
# Confirmed from CSV: micro=10, bench=500, pilot=2000
SCALE_COSTS = {0: 10.0, 1: 500.0, 2: 2000.0}
 
# Noise variances (sigma^2) — pooled from empirical repeat data
# micro:  pooled var of baseline (0.000405) and hotspot (0.000196) repeats
# bench:  conservative estimate — only 2 hotspot reps available
# pilot:  assumed lower than bench per project spec
SCALE_NOISE = {
    0: 0.000300,   # micro  — pooled empirical
    1: 0.003600,   # bench  — conservative estimate
    2: 0.000100,   # pilot  — spec assumption
}
 


#
#   micro-micro  = 1.00  (identity)
#   micro-bench  = 0.20  (micro Y ~30x lower than bench, weak signal)
#   micro-pilot  = 0.15  (micro Y ~13x lower than pilot, weakest)
#   bench-bench  = 1.00  (identity)
#   bench-pilot  = 0.90  (bench/pilot ratio 2.3–2.5, consistent → strong)
#   pilot-pilot  = 1.00  (identity)
B = np.array([
    [1.00, 0.20, 0.15],   # micro
    [0.20, 1.00, 0.90],   # bench
    [0.15, 0.90, 1.00],   # pilot
])

class MultiFidelityKernel(RBF):
    """
    RBF kernel on the 5 recipe dimensions, multiplied by a scale-correlation
    matrix B on the 6th (scale index) dimension.
    """
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
 

def expected_improvement(mean: float, std: float, best_f: float) -> float:
    if std <= 1e-6:
        return 0.0
    z  = (mean - best_f) / std
    ei = (mean - best_f) * norm.cdf(z) + std * norm.pdf(z)
    return float(max(0.0, ei))
 
 

def select_next_experiment(
    gp_model: GaussianProcessRegressor,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    n_candidates: int = 3000,  
) -> tuple[np.ndarray, int]:
    """
    For each scale, evaluate EI at that scale (not always pilot=2) and
    normalise by cost.  Previously the GP was always queried with scale=2
    regardless of which scale was being evaluated — this made micro win
    every time because EI was identical across scales but cost was lowest.
 
    Now the GP is queried at the actual candidate scale so the model can
    genuinely distinguish what it expects from a micro vs bench vs pilot run.
    """
    pilot_mask = X_train[:, 5] == 2
    if np.any(pilot_mask):
        best_f = float(np.max(Y_train[pilot_mask]))
    else:
        best_f = float(np.max(Y_train))
 
    best_utility = -np.inf
    best_recipe  = None
    best_scale   = None
 

    random_recipes = np.random.uniform(
        RECIPE_MIN, RECIPE_MAX, size=(n_candidates, 5)
    )
 
    for scale_idx in [0, 1, 2]:
        X_cand = np.column_stack([
            random_recipes,
            np.full(n_candidates, scale_idx, dtype=float),
        ])
        means, stds = gp_model.predict(X_cand, return_std=True)
        # ─────────────────────────────────────────────────────────────────
 
        for recipe, mean, std in zip(random_recipes, means, stds):
            ei      = expected_improvement(float(mean), float(std), best_f)
            utility = ei / SCALE_COSTS[scale_idx]
 
            if utility > best_utility:
                best_utility = utility
                best_recipe  = recipe.copy()
                best_scale   = scale_idx
 
    return best_recipe, best_scale
 #MOCKAPI FUNCTION

def experiment(recipe: np.ndarray, scale: int) -> tuple[float, float]:
    r = np.clip(recipe, RECIPE_MIN, RECIPE_MAX)
    recipe_dict = {
        "T":  float(r[0]),
        "pH": float(r[1]),
        "F1": float(r[2]),
        "F2": float(r[3]),
        "F3": float(r[4]),
    }
    print(
        f"  API call -> scale={SCALE_MAPPING[scale]:<5s} | "
        + "  ".join(f"{k}={v:.3f}" for k, v in recipe_dict.items()),
        flush=True,
    )
    resp = client.run(SCALE_MAPPING[scale], **recipe_dict)
    y    = float(resp["Y"])
    cost = SCALE_COSTS[scale]
    basic_client.time.sleep(0.5)
    return y, cost



MOCK_MODE = True   # flip to False for real runs

def experiment_mock(recipe: np.ndarray, scale: int) -> tuple[float, float]:
    HOTSPOT = np.array([35.5, 6.5, 1.0, 1.0, 1.0])
    dist    = np.linalg.norm(recipe - HOTSPOT)
    base = {0: 0.14, 1: 3.52, 2: 1.44}
    peak = {0: 0.18, 1: 5.42, 2: 2.36}
    t    = max(0.0, 1.0 - dist / 10.0)
    y    = base[scale] + t * (peak[scale] - base[scale])
    y   += np.random.normal(0, np.sqrt(SCALE_NOISE[scale]))
    y    = max(0.0, y)
    cost = SCALE_COSTS[scale]
    print(f"  MOCK -> scale={SCALE_MAPPING[scale]:<5s}  Y={y:.4f}  cost={cost}€")
    return y, cost


if MOCK_MODE:
    experiment = experiment_mock


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
        ([35.5, 6.5, 1.0, 1.0, 1.0],               2,   2.35823),  # pilot ← most valuable
    ]
    # ─────────────────────────────────────────────────────────────────────
 
 
    history: list[dict] = []
 
    X_list, Y_list = [], []
    for recipe, scale, y in historical_seed:
        x_row = np.append(recipe, scale)
        X_list.append(x_row)
        Y_list.append(y)
        history.append({
            "T": recipe[0], "pH": recipe[1],
            "F1": recipe[2], "F2": recipe[3], "F3": recipe[4],
            "scale":           scale,
            "observed_Y":      y,
            "cost_eur":        0.0,    # already paid — don't count against budget
            "cumulative_cost": 0.0,
            "source":          "historical_seed",
        })
    # ─────────────────────────────────────────────────────────────────────
 
    X_train = np.array(X_list)
    Y_train = np.array(Y_list)
    total_spent = 0.0
 
    print(f"Seeded GP with {len(X_list)} historical observations "
          f"(micro={sum(1 for _,s,_ in historical_seed if s==0)}, "
          f"bench={sum(1 for _,s,_ in historical_seed if s==1)}, "
          f"pilot={sum(1 for _,s,_ in historical_seed if s==2)})")
 
    # Build GP
    mf_kernel  = MultiFidelityKernel(length_scale=5.0)
    alpha_noise = np.array([SCALE_NOISE[int(s)] for s in X_train[:, 5]])
    gp = GaussianProcessRegressor(
        kernel=mf_kernel,
        alpha=alpha_noise,
        n_restarts_optimizer=5,   
        normalize_y=True,
    )
 
    print("\n--- Launching Cost-Aware BO Loop ---")
    while total_spent < BUDGET_LIMIT:
        remaining = BUDGET_LIMIT - total_spent
        print(f"\nBudget: {total_spent:.0f} / {BUDGET_LIMIT:.0f} EUR  "
              f"(remaining: {remaining:.0f}€)", flush=True)
 
        # Skip a scale if we can't afford it
        affordable = [s for s in [0, 1, 2] if SCALE_COSTS[s] <= remaining]
        if not affordable:
            print("No affordable scale remaining — stopping.")
            break
 
        # Recompute alpha each iteration (X_train grows)
        gp.alpha = np.array([SCALE_NOISE[int(s)] for s in X_train[:, 5]])
        gp.fit(X_train, Y_train)
 
        next_recipe, next_scale = select_next_experiment(gp, X_train, Y_train)
 
        # Don't select a scale we can't afford
        if next_scale not in affordable:
            next_scale = max(affordable)   # fall back to most informative affordable scale
            print(f"  [scale downgraded to {SCALE_MAPPING[next_scale]} — budget constraint]")
 
        y, cost = experiment(next_recipe, next_scale)
        total_spent += cost
 
        X_train = np.vstack([X_train, np.append(next_recipe, next_scale)])
        Y_train = np.append(Y_train, y)
 
        pilot_mask = X_train[:, 5] == 2
        best_pilot = float(np.max(Y_train[pilot_mask])) if np.any(pilot_mask) else 0.0
 

        history.append({
            "T":               float(next_recipe[0]),
            "pH":              float(next_recipe[1]),
            "F1":              float(next_recipe[2]),
            "F2":              float(next_recipe[3]),
            "F3":              float(next_recipe[4]),
            "scale":           next_scale,
            "observed_Y":      y,
            "cost_eur":        cost,
            "cumulative_cost": total_spent,
            "source":          "bo_loop",
        })
 
        print(
            f"  -> Y={y:.4f}  cost={cost:.0f}€  "
            f"total={total_spent:.0f}€  best_pilot={best_pilot:.4f}",
            flush=True,
        )
 
    # ── Save results ──────────────────────────────────────────────────────
    csv_filename = "bo_campaign_history.csv"
    fieldnames   = ["T", "pH", "F1", "F2", "F3", "scale",
                    "observed_Y", "cost_eur", "cumulative_cost", "source"]
 
    with open(csv_filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)   # Bug 3 fix: all rows present, no zip mismatch
 
    # ── Final summary ─────────────────────────────────────────────────────
    pilot_mask = X_train[:, 5] == 2
    max_pilot_Y = float(np.max(Y_train[pilot_mask])) if np.any(pilot_mask) else 0.0
    n_by_scale  = {SCALE_MAPPING[s]: int(np.sum(X_train[:, 5] == s)) for s in [0, 1, 2]}
 
    print("\n" + "=" * 55)
    print(f"Campaign complete — results saved to {csv_filename}")
    print(f"Runs by scale:   {n_by_scale}")
    print(f"Total cost:      {total_spent:.0f} EUR")
    print(f"Best pilot Y:    {max_pilot_Y:.4f} g/L   (Felix baseline: 14.0)")
    print("=" * 55)
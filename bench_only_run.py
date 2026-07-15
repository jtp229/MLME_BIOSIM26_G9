import csv
import numpy as np
from scipy.stats import norm, qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import time
from scipy.optimize import minimize

import basic_client

client = basic_client.BioreactorClient(basic_client.BASE_URL)
client.login(basic_client.USER, basic_client.PASSWORD)

RECIPE_MIN = np.array([20.0,  3.0, 0.0, 0.0, 0.0])
RECIPE_MAX = np.array([60.0,  9.5, 2.0, 2.0, 2.0])

SCALE_MAPPING = {0: "micro", 1: "bench", 2: "pilot"}
TARGET_PILOT_Y: float | None = None

SCALE_COSTS = {0: 10.0, 1: 500.0, 2: 2000.0}
SCALE_NOISE = {
    0: 0.000365,   # micro (calculated from 50 runs)
    1: 0.003797,   # bench (calculated from 50 runs)
    2: 0.000024,   # pilot (calculated from 50 runs)
}

B = np.array([
    [1.00, 0.20, 0.15],
    [0.20, 1.00, 0.90],
    [0.15, 0.90, 1.00],
])

class MultiFidelityKernel(RBF):
    def __call__(self, X, Y=None, eval_gradient=False):
        X_recipe = X[:, :5]
        Y_recipe = Y[:, :5] if Y is not None else None

        if eval_gradient:
            rbf_matrix, rbf_gradient = super().__call__(X_recipe, Y_recipe, eval_gradient=True)
        else:
            rbf_matrix = super().__call__(X_recipe, Y_recipe, eval_gradient=False)

        # High-performance vectorization replacing the nested loops
        X_scale = X[:, 5].astype(int)
        Y_scale = Y[:, 5].astype(int) if Y is not None else X_scale
        b_matrix = B[X_scale[:, np.newaxis], Y_scale]

        if eval_gradient:
            return rbf_matrix * b_matrix, rbf_gradient * b_matrix[:, :, np.newaxis]
        return rbf_matrix * b_matrix


def expected_improvement(mean: float, std: float, best_f: float, xi: float = 0.05) -> float:
    if std <= 1e-6:
        return 0.0
    z  = (mean - best_f - xi) / std
    ei = (mean - best_f - xi) * norm.cdf(z) + std * norm.pdf(z)
    return float(max(0.0, ei))


def expected_target_utility(mean: float, std: float, target: float) -> float:
    return -(((mean - target) ** 2) + (std ** 2))


def transform_X(X: np.ndarray) -> np.ndarray:
    X_scaled = X.copy()
    X_scaled[:, :5] = (X[:, :5] - RECIPE_MIN) / (RECIPE_MAX - RECIPE_MIN)
    return X_scaled


def select_next_experiment(
    gp_model: GaussianProcessRegressor,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    n_candidates: int = 1000,
) -> tuple[np.ndarray, int]:
    
    # 1. Determine best observed value
    pilot_mask = X_train[:, 5] == 2
    if TARGET_PILOT_Y is None:
        if np.any(pilot_mask):
            best_pilot_f = float(np.max(Y_train[pilot_mask]))
        else:
            X_train_scaled = transform_X(X_train)
            X_train_pilot = X_train_scaled.copy()
            X_train_pilot[:, 5] = 2
            best_pilot_f = float(np.max(gp_model.predict(X_train_pilot)))
    else:
        best_pilot_f = float("nan")

    # 2. Find the best starting point via random sampling
    random_recipes_scaled = np.random.uniform(0.0, 1.0, size=(n_candidates, 5))
    X_cand_pilot = np.column_stack([random_recipes_scaled, np.full(n_candidates, 2.0)])
    means, stds = gp_model.predict(X_cand_pilot, return_std=True)
    
    if TARGET_PILOT_Y is None:
        acqs = [expected_improvement(m, s, best_pilot_f, xi=0.01) for m, s in zip(means, stds)]
    else:
        acqs = [expected_target_utility(m, s, TARGET_PILOT_Y) for m, s in zip(means, stds)]
    
    best_idx = np.argmax(acqs)
    start_recipe = random_recipes_scaled[best_idx]

    # 3. Local Refinement using L-BFGS-B
    def neg_acq(x_scaled):
        x_full = np.column_stack([x_scaled.reshape(1, -1), [[2.0]]])
        m, s = gp_model.predict(x_full, return_std=True)
        
        m_val = m.item()
        s_val = s.item()
        
        if TARGET_PILOT_Y is None:
            return -expected_improvement(m_val, s_val, best_pilot_f, xi=0.01)
        return -expected_target_utility(m_val, s_val, TARGET_PILOT_Y)

    # Run the gradient descent on the surrogate model
    res = minimize(
        neg_acq, 
        x0=start_recipe, 
        method='L-BFGS-B', 
        bounds=[(0, 1)] * 5
    )
    best_recipe_scaled = res.x

    # 4. Scale selection 
    X_eval_scales = np.array([np.append(best_recipe_scaled, s) for s in [0, 1, 2]])
    _, stds_at_scales = gp_model.predict(X_eval_scales, return_std=True)
    
    pilot_std = float(stds_at_scales[2])
    if pilot_std > 0.97: best_scale = 0
    elif pilot_std > 0.44: best_scale = 1
    else: best_scale = 2

    best_recipe = best_recipe_scaled * (RECIPE_MAX - RECIPE_MIN) + RECIPE_MIN
    return best_recipe, best_scale


def experiment(recipe: np.ndarray, scale: int) -> tuple[float, float, int]:
    # --- ONLY BENCH INTERCEPT ---
    # If it's the LHS phase (history is short), let it use Micro (0).
    # Once the BO loop starts, force everything to Bench (1) until the final Pilot pass.
    if len(history) >= 100 and total_spent < (BUDGET_LIMIT - PILOT_RESERVE):
        scale = 1
    # ----------------------------
    
    r = np.clip(recipe, RECIPE_MIN, RECIPE_MAX)
    recipe_dict = {
        "T":  float(r[0]), "pH": float(r[1]), "F1": float(r[2]), "F2": float(r[3]), "F3": float(r[4]),
    }
    print(f"  API call -> scale={SCALE_MAPPING[scale]:<5s} | " + "  ".join(f"{k}={v:.3f}" for k, v in recipe_dict.items()), flush=True)
    resp = client.run(SCALE_MAPPING[scale], **recipe_dict)
    y    = float(resp["Y"])
    cost = SCALE_COSTS[scale]
    basic_client.time.sleep(1.0)
    return y, cost, scale


MOCK_MODE = False
if __name__ == "__main__":
    BUDGET_LIMIT = 15000.0
    PILOT_RESERVE = 2000.0     
    N_INITIAL_SAMPLES = 100     
    INITIAL_SCALE = 0          
    
    history: list[dict] = []
    X_list, Y_list = [], []
    total_spent = 0.0
    
    print(f"--- Generating {N_INITIAL_SAMPLES} initial points using Latin Hypercube Sampling ---")
    sampler = qmc.LatinHypercube(d=5)
    lhs_sample = sampler.random(n=N_INITIAL_SAMPLES)
    initial_recipes = qmc.scale(lhs_sample, RECIPE_MIN, RECIPE_MAX)
    
    for recipe in initial_recipes:
        if total_spent + SCALE_COSTS[INITIAL_SCALE] > BUDGET_LIMIT:
            print("Budget exceeded during LHS initialization.")
            break
            
        y, cost, executed_scale = experiment(recipe, INITIAL_SCALE)
        total_spent += cost
        X_list.append(np.append(recipe, executed_scale))
        Y_list.append(y)
        
        history.append({
            "T": float(recipe[0]), "pH": float(recipe[1]), "F1": float(recipe[2]), "F2": float(recipe[3]), "F3": float(recipe[4]),
            "scale": executed_scale, "observed_Y": y, "cost_eur": cost, "cumulative_cost": total_spent, "source": "lhs_seed",
        })

    X_train = np.array(X_list)
    Y_train = np.array(Y_list)

    # Stabilized length_scale lower bound to prevent micro-spires
    mf_kernel  = MultiFidelityKernel(length_scale=0.2, length_scale_bounds=(0.15, 1.5))
    gp = GaussianProcessRegressor(
        kernel=mf_kernel,
        alpha=np.array([SCALE_NOISE[int(s)] for s in X_train[:, 5]]),
        n_restarts_optimizer=5,   
        normalize_y=True,
    )

    print("\n--- Launching Cost-Aware BO Loop ---")
    while total_spent < BUDGET_LIMIT:
        remaining = BUDGET_LIMIT - total_spent
        print(f"\nBudget: {total_spent:.0f} / {BUDGET_LIMIT:.0f} EUR  (remaining: {remaining:.0f}€)", flush=True)

        gp.alpha = np.array([SCALE_NOISE[int(s)] for s in X_train[:, 5]])

        if remaining <= PILOT_RESERVE:
            print("\n!!! Final 2000€ reached: Commencing target optimization for the Pilot scale !!!")
            
            X_train_scaled = transform_X(X_train)
            gp.fit(X_train_scaled, Y_train)
            
            exploitation_candidates = 5000
            random_recipes_scaled = np.random.uniform(0.0, 1.0, size=(exploitation_candidates, 5))
            X_cand_pilot = np.column_stack([
                random_recipes_scaled,
                np.full(exploitation_candidates, 2, dtype=float), 
            ])
            
            predicted_means = gp.predict(X_cand_pilot)
            best_idx = np.argmax(predicted_means)
            
            next_recipe = random_recipes_scaled[best_idx] * (RECIPE_MAX - RECIPE_MIN) + RECIPE_MIN
            next_scale = 2
            print(f"  [Exploitation complete. Maximum predicted Pilot yield found at chosen recipe]")
            
        else:
            affordable = [s for s in [0, 1, 2] if SCALE_COSTS[s] <= (remaining - PILOT_RESERVE)]
            if not affordable:
                affordable = [1]
                
            X_train_scaled = transform_X(X_train)
            gp.fit(X_train_scaled, Y_train)

            next_recipe, next_scale = select_next_experiment(gp, X_train, Y_train)

            if next_scale not in affordable:
                next_scale = max(affordable)
                print(f"  [scale downgraded to {SCALE_MAPPING[next_scale]} — preserving pilot budget reserve]")

        # Capturing returned scale directly to resolve variable scoping mismatch
        y, cost, next_scale = experiment(next_recipe, next_scale)
        total_spent += cost

        X_train = np.vstack([X_train, np.append(next_recipe, next_scale)])
        Y_train = np.append(Y_train, y)

        pilot_mask = X_train[:, 5] == 2
        best_overall = float(np.max(Y_train))
        best_pilot = float(np.max(Y_train[pilot_mask])) if np.any(pilot_mask) else 0.0

        history.append({
            "T": float(next_recipe[0]), "pH": float(next_recipe[1]), "F1": float(next_recipe[2]), "F2": float(next_recipe[3]), "F3": float(next_recipe[4]),
            "scale": next_scale, "observed_Y": y, "cost_eur": cost, "cumulative_cost": total_spent, "source": "bo_loop" if remaining > PILOT_RESERVE else "final_pilot_exploitation",
        })

        print(f"  -> Y={y:.4f}  cost={cost:.0f}€  total={total_spent:.0f}€  best={best_overall:.4f}  best_pilot={best_pilot:.4f}", flush=True)

    csv_filename = "bo_only_bench.csv"
    fieldnames   = ["T", "pH", "F1", "F2", "F3", "scale", "observed_Y", "cost_eur", "cumulative_cost", "source"]
    with open(csv_filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)   

    print("\n" + "=" * 55)
    print(f"Campaign complete — results saved to {csv_filename}")
    print(f"Total cost:      {total_spent:.0f} EUR")
    print(f"Best overall Y:  {float(np.max(Y_train)):.4f} g/L")
    print(f"Best pilot Y:    {float(np.max(Y_train[X_train[:, 5] == 2])) if np.any(X_train[:, 5] == 2) else 0.0:.4f} g/L")
    print("=" * 55)
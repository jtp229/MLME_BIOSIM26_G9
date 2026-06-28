import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Hyperparameter
import basic_client
import csv

client = basic_client.BioreactorClient(basic_client.BASE_URL)
client.login(basic_client.USER, basic_client.PASSWORD)

# Recipe variables: [T, pH, F1, F2, F3]
RECIPE_MIN = np.array([20.0, 3.0, 0.0, 0.0, 0.0])
RECIPE_MAX = np.array([60.0, 9.5, 2.0, 2.0, 2.0])

SCALE_MAPPING = {
    0: "micro",   # Microplate
    1: "bench",   # Benchtop
    2: "pilot"    # Pilot
}

SCALE_COSTS = {0: 10.0, 1: 500.0, 2: 2000.0}       # Micro, Bench, Pilot
SCALE_NOISE = {
     0: 0.00033,  # Empirical Microplate noise
     1: 0.00728,  # Empirical Benchtop noise
     2: 0.00010   # Pilot noise assumption based on bench scale
}  # Variance (\sigma^2). 
B = np.array([
    [1.0, 0.5, 0.3],  # Microplate correlations
    [0.5, 1.0, 0.8],  # Benchtop correlations
    [0.3, 0.8, 1.0]   # Pilot correlations
])


def experiment(recipe, scale):
    r = np.clip(recipe, RECIPE_MIN, RECIPE_MAX)
    api_scale_string = SCALE_MAPPING[scale]

    recipe_dict = {
        "T": float(r[0]),
        "pH": float(r[1]),
       "F1": float(r[2]),
        "F2": float(r[3]),
       "F3": float(r[4])
    }

    print(f"Calling API -> Scale: {api_scale_string} | Recipe parameters: {recipe_dict}")
    api_response = client.run(api_scale_string, **recipe_dict)

    observed_y = float(api_response["Y"])
    cost = SCALE_COSTS[scale]

    basic_client.time.sleep(0.5)
    return observed_y, cost


class multifidelity(RBF):
    def __call__(self, X, Y=None, eval_gradient=False):
        if eval_gradient:
            rbf_matrix, rbf_gradient = super().__call__(X[:, :5], None if Y is None else Y[:, :5], eval_gradient=True)
        else:
            rbf_matrix = super().__call__(X[:, :5], None if Y is None else Y[:, :5], eval_gradient=False)
            
        num_rows_X = X.shape[0]
        num_cols_Y = Y.shape[0] if Y is not None else num_rows_X

        b_matrix = np.zeros((num_rows_X, num_cols_Y))

        for i in range(num_rows_X):
            for j in range(num_cols_Y):
                scale_i = int(X[i, 5])
                scale_j = int(Y[j, 5]) if Y is not None else int(X[j, 5])
                b_matrix[i, j] = B[scale_i, scale_j]
        if eval_gradient:
            b_gradient = rbf_gradient * b_matrix[:, :, np.newaxis]
            return rbf_matrix * b_matrix, b_gradient
        else:
            return rbf_matrix * b_matrix


def get_expected_improvement(mean, std, best_f):
    if std <= 1e-6:
        return 0.0  
    z = (mean - best_f) / std
    ei = (mean - best_f) * norm.cdf(z) + std * norm.pdf(z)
    return ei


def select_next_experiment(gp_model, X_train, Y_train):
    pilot_indices = X_train[:, 5] == 2
    best_f = np.max(Y_train[pilot_indices]) if np.any(pilot_indices) else np.max(Y_train)  

    ValuePerEuro = -float('inf')
    best_recipe = None
    best_scale = None

    num_candidates = 1000  
    random_recipes = np.random.uniform(RECIPE_MIN, RECIPE_MAX, size=(num_candidates, 5))

    for scale_idx in [0, 1, 2]:
        for recipe in random_recipes:
            pilot_context_x = np.append(recipe, 2).reshape(1, -1)  
            mean, std = gp_model.predict(pilot_context_x, return_std=True)
            raw_ei = get_expected_improvement(mean[0], std[0], best_f)
            cost_aware_utility = raw_ei / SCALE_COSTS[scale_idx]  

            if cost_aware_utility > ValuePerEuro:
                ValuePerEuro = cost_aware_utility
                best_recipe = recipe
                best_scale = scale_idx
    return best_recipe, best_scale


if __name__ == "__main__":
    X_list, Y_list = [], []
    total_spent = 0.0
    BUDGET_LIMIT = 15000.0  
    
    print("--- Running Random Initial Seeds (Restored State) ---")
    num_initial_seeds = 3
    for _ in range(num_initial_seeds):
        random_seed_recipe = np.random.uniform(RECIPE_MIN, RECIPE_MAX)
        y, cost = experiment(random_seed_recipe, 0)
        X_list.append(np.append(random_seed_recipe, 0))
        Y_list.append(y)
        total_spent += cost

    X_train = np.array(X_list)
    Y_train = np.array(Y_list)

    # UNOPTIMIZED: Allowed hyperparameter optimization to fluctuate bounds freely
    mf_kernel = multifidelity(length_scale=5.0) 
    alpha_noise = np.array([SCALE_NOISE[int(s)] for s in X_train[:, 5]])
    
    # UNOPTIMIZED: Internal hyperparameter tuning enabled (Heavy calculation active)
    gp = GaussianProcessRegressor(kernel=mf_kernel, alpha=alpha_noise, n_restarts_optimizer=0)
    
    history_cost = [total_spent]
    history_yield = [0.0]
    history_scales = [0]

    print("\n--- Launching Cost-Aware Optimization Loop ---")
    while total_spent < BUDGET_LIMIT:
        print(f"\nBudget Status: {total_spent:.1f} / {BUDGET_LIMIT} EUR spent.")
        
        gp.fit(X_train, Y_train)
        next_recipe, next_scale = select_next_experiment(gp, X_train, Y_train)
        
        y, cost = experiment(next_recipe, next_scale)
        total_spent += cost
        
        X_train = np.vstack([X_train, np.append(next_recipe, next_scale)])
        Y_train = np.append(Y_train, y)
        
        alpha_noise = np.array([SCALE_NOISE[int(s)] for s in X_train[:, 5]])
        gp.alpha = alpha_noise
        
        pilot_runs = X_train[:, 5] == 2
        current_best_pilot = np.max(Y_train[pilot_runs]) if np.any(pilot_runs) else 0.0
        
        history_cost.append(total_spent)
        history_yield.append(current_best_pilot)
        history_scales.append(next_scale)
        
    # SAVING ALL DATA INSTEAD OF FALLBACK VALUES
    csv_filename = "bo_campaign_history.csv"
    with open(csv_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["T", "pH", "F1", "F2", "F3", "Scale", "Observed_Yield", "Cumulative_Cost"])
        for x, y, c in zip(X_train, Y_train, history_cost):
            writer.writerow([x[0], x[1], x[2], x[3], x[4], int(x[5]), y, c])

    print("\n=======================================================")
    print(f"Optimization trace saved to {csv_filename}")
    pilot_executed = X_train[:, 5] == 2
    max_yield = np.max(Y_train[pilot_executed]) if np.any(pilot_executed) else 0.0
    print(f"Max Pilot Yield Achieved: {max_yield:.4f} g/L")
    print("=======================================================")
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
     1: 0.00728,    # Empirical Benchtop noise
     2: 0.00010    # Pilot noise assumption based on bench scale
}  # Variance (\sigma^2). 
B = np.array([
    [1.0, 0.5, 0.3],  # Microplate correlations
    [0.5, 1.0, 0.8],  # Benchtop correlations
    [0.3, 0.8, 1.0]   # Pilot correlations
]

)


def expirement(recipe,scale):
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
        rbf_matrix = super().__call__(X[:, :5], None if Y is None else Y[:, :5], eval_gradient=False)
        #Only take the first 5 columns of x and y for the RBF Kernel so that the kernel is only applied to the recipe variables. The last column of x and y is the scale variable, which is not used in the RBF kernel.
      
        num_rows_X = X.shape[0] # Find dimensions of rows and columns (N and M)
        num_cols_Y = Y.shape[0] if Y is not None else num_rows_X

        b_matrix = np.zeros((num_rows_X, num_cols_Y)) #Create grid of zeros matching rbf_matrix size

        #fill grid
        for i in range(num_rows_X):
            for j in range(num_cols_Y):
               # Extract the scale index (0, 1, or 2) for experiment i and experiment j
                scale_i = int(X[i, 5])
                scale_j = int(Y[j, 5]) if Y is not None else int(X[j, 5])
                b_matrix[i, j] = B[scale_i, scale_j] # Fill the grid with the corresponding value from the B matrix

        return rbf_matrix * b_matrix  # Element-wise multiplication of the RBF matrix and the B matrix
            
def get_expected_improvement(mean, std, best_f):
        if std <= 1e-6:
            return 0.0  # if standard deviation is too small, return 0 to avoid wasting money on repeat test
        # Calculate the expected improvement
        z = (mean - best_f) / std
        ei = (mean - best_f) * norm.cdf(z) + std * norm.pdf(z) #look at probability of average perfomance and then explore by multiplying std by probability density function of z 
        return ei
    
def select_next_experiment(gp_model, X_train, Y_train):
        pilot_indices = X_train[:, 5] == 2
        best_f = np.max(Y_train[pilot_indices]) if np.any(pilot_indices) else np.max(Y_train)  # Use the best observed value from pilot scale experiments, or overall if none exist

        ValuePerEuro = -float('inf') #start at -inf so that any new value will become record holder
        best_recipe = None
        best_scale = None

        num_candidates = 1000  # Number of random candidates to sample
        random_recipes = np.random.uniform(RECIPE_MIN, RECIPE_MAX, size=(num_candidates, 5))

        for scale_idx in [0,1,2]:
            for recipe in random_recipes:
                pilot_context_x = np.append(recipe, 2).reshape(1, -1)  
                mean, std = gp_model.predict(pilot_context_x, return_std=True)
                raw_ei = get_expected_improvement(mean[0], std[0], best_f)
                cost_aware_utility = raw_ei / SCALE_COSTS[scale_idx]  # Normalize by cost

                if cost_aware_utility > ValuePerEuro:
                    ValuePerEuro = cost_aware_utility
                    best_recipe = recipe
                    best_scale = scale_idx
        return best_recipe, best_scale



            
if __name__ == "__main__":
    X_list, Y_list = [], []
    total_spent = 0.0
    BUDGET_LIMIT = 15000.0  # Felix budget
    

    print("--- Running Initial Seeds ---")
    for i in range(3):
        # Pick a random recipe inside the bounds
        init_recipe = np.random.uniform(RECIPE_MIN, RECIPE_MAX)
        
        # Execute it on the cheap Microplate
        y, cost = expirement(init_recipe, 0)
        
        X_list.append(np.append(init_recipe, 0))
        Y_list.append(y)
        total_spent += cost
        print(f"Seed {i+1}/3 completed on Micro scale. Spent: {total_spent} EUR")

    # Convert python lists to numpy matrices for sklearn
    X_train = np.array(X_list)
    Y_train = np.array(Y_list)

   
    # Instantiate multi-fidelity kernel 
    mf_kernel = multifidelity(length_scale=15.0)
    
    # Generate the noise tracker array 
    alpha_noise = np.array([SCALE_NOISE[int(s)] for s in X_train[:, 5]])
    
    # Define the core scikit-learn regressor
    gp = GaussianProcessRegressor(kernel=mf_kernel, alpha=alpha_noise, n_restarts_optimizer=0)
    
    # Lists to track history for plotting
    history_cost = [total_spent]
    history_yield = [0.0]
    history_scales = [0]


    print("\n---  Starting Optimization Loop ---")
    while total_spent < BUDGET_LIMIT:
        print(f"\nBudget Status: {total_spent:.1f} / {BUDGET_LIMIT} EUR spent.")
        
        #  Fit the model on all historical observations
        gp.fit(X_train, Y_train)
        
        # Use acquisition functions to choose the next best coordinates
        next_recipe, next_scale = select_next_experiment(gp, X_train, Y_train)
        
        # Call live server API 
        y, cost = expirement(next_recipe, next_scale)
        total_spent += cost
        
        #  Append new data to the arrays
        X_train = np.vstack([X_train, np.append(next_recipe, next_scale)])
        Y_train = np.append(Y_train, y)
        
        # Dynamically adjust the alpha parameter array for the fresh row entry
        alpha_noise = np.array([SCALE_NOISE[int(s)] for s in X_train[:, 5]])
        gp.alpha = alpha_noise
        
        # Extract current highest record found on the pilot scale specifically
        pilot_runs = X_train[:, 5] == 2
        current_best_pilot = np.max(Y_train[pilot_runs]) if np.any(pilot_runs) else 0.0
        
        # Track history logs
        history_cost.append(total_spent)
        history_yield.append(current_best_pilot)
        history_scales.append(next_scale)

    csv_filename = "bo_campaign_history.csv"
    with open(csv_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Cost_Spent", "Best_Pilot_Yield", "Chosen_Scale"])
        for c, y, s in zip(history_cost, history_yield, history_scales):
            writer.writerow([c, y, s])

    print("\n=======================================================")
    print(f" History saved to {csv_filename}")
    print(f"Highest Pilot Scale Yield Discovered: {np.max(Y_train[X_train[:, 5] == 2]):.4f} g/L")
    print("=======================================================")
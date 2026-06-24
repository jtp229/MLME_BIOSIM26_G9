import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Recipe variables: [T, pH, F1, F2, F3]
RECIPE_MIN = np.array([20.0, 3.0, 0.0, 0.0, 0.0])
RECIPE_MAX = np.array([60.0, 9.5, 2.0, 2.0, 2.0])


SCALE_COSTS = {0: 10.0, 1: 500.0, 2: 2000.0}       # Micro, Bench, Pilot
SCALE_NOISE = {0: 1.5**2, 1: 0.5**2, 2: 0.05**2}  # Variance (\sigma^2). Placeholder variables until Ian finishes exploration
#scale similarity matrix
B = np.array([
    [1.0, 0.5, 0.3],  # Microplate correlations
    [0.5, 1.0, 0.8],  # Benchtop correlations
    [0.3, 0.8, 1.0]   # Pilot correlations
]

)

#mock process without Ian data

def expirement(recipe,scale):
    r = np.clip(recipe, RECIPE_MIN, RECIPE_MAX)
    base_yield = 16.0 - 0.015*(r[0]-37)**2 - 0.6*(r[1]-7.2)**2 - 0.3*(r[2]-1.0)**2 #when variables move away from optimum values yield decreases

    if scale == 0:
        bias, noise_sd = -2.5, 1.5
    elif scale == 1:
        bias, noise_sd = -0.8, 0.5
    else:
        bias, noise_sd = 0.0, 0.05
    observed_y = max(0.0, base_yield + bias + np.random.normal(0, noise_sd))
    return observed_y, SCALE_COSTS[scale]



import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
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
SCALE_NOISE = {0: 1.5**2, 1: 0.5**2, 2: 0.05**2}  # Variance (\sigma^2). Placeholder variables until Ian finishes analysis of noise.
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

   
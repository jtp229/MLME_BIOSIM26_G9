import time
import requests
import numpy as np
from scipy.stats import qmc

# Credentials
USER = "group09"
PASSWORD = "3v13c-n3qcr-y3qj9"
BASE_URL = "https://mlme26biosim.org"

# Bounds defined by the project description
RECIPE_MIN = np.array([20.0,  3.0, 0.0, 0.0, 0.0])
RECIPE_MAX = np.array([60.0,  9.5, 2.0, 2.0, 2.0])

SCALE_MAPPING = {0: "micro", 1: "bench", 2: "pilot"}

class BioreactorClient:
    """Cookie-based session wrapper around the lab's REST API."""
    def __init__(self, base_url: str = BASE_URL):
        self.s = requests.Session()
        self.base = base_url.rstrip("/")

    def login(self, user: str, password: str) -> None:
        r = self.s.post(
            f"{self.base}/api/login",
            json={"user": user, "password": password},
            timeout=15,
        )
        r.raise_for_status()

    def _csrf(self) -> str:
        token = self.s.cookies.get("mlme26_csrf")
        if not token:
            raise RuntimeError("no CSRF cookie set — call login() first")
        return token

    def run(self, scale: str, T: float, pH: float,
            F1: float, F2: float, F3: float) -> dict:
        payload = {
            "scale": scale,
            "recipe": {"T": T, "pH": pH, "F1": F1, "F2": F2, "F3": F3},
        }
        for attempt in range(8):
            try:
                r = self.s.post(
                    f"{self.base}/api/run", json=payload,
                    headers={"X-CSRF-Token": self._csrf()},
                    timeout=60,
                )
                if r.status_code == 429 or r.status_code >= 500:
                    time.sleep(2.5 ** attempt)
                    continue
                if r.status_code == 402:
                    raise RuntimeError(f"Budget exhausted: {r.json().get('detail')}")
                r.raise_for_status()
                return r.json()
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                time.sleep(2.5 ** attempt)
        raise RuntimeError("Too many failed network attempts")


if __name__ == "__main__":
    client = BioreactorClient(BASE_URL)
    client.login(USER, PASSWORD)

    # 1. Generate 5 widely spread out recipes using Latin Hypercube Sampling
    print("--- Generating 5 diverse recipe templates ---")
    sampler = qmc.LatinHypercube(d=5)
    samples = sampler.random(n=5)
    test_recipes = qmc.scale(samples, RECIPE_MIN, RECIPE_MAX)

    # Matrix to hold yields: 5 rows (recipes), 3 columns (Micro, Bench, Pilot)
    yield_matrix = np.zeros((5, 3))

    print("\n--- Starting Live Multi-Scale Evaluation ---")
    for idx, recipe in enumerate(test_recipes):
        print(f"\nEvaluating Recipe Template {idx+1}/5...")
        recipe_dict = {
            "T": float(recipe[0]), 
            "pH": float(recipe[1]), 
            "F1": float(recipe[2]), 
            "F2": float(recipe[3]), 
            "F3": float(recipe[4])
        }
        
        # Test the exact same recipe at all three scales to see how they co-vary
        for scale_idx, scale_name in SCALE_MAPPING.items():
            print(f"  -> Testing scale [{scale_name.upper()}]...", end="", flush=True)
            start = time.time()
            res = client.run(scale_name, **recipe_dict)
            yield_matrix[idx, scale_idx] = float(res["Y"])
            print(f" Done. Y = {res['Y']:.4f} ({time.time()-start:.1f}s)")
            time.sleep(0.5)

    # 2. Compute the Pearson Correlation Matrix across the columns
    # Transpose from (5,3) to (3,5) because np.corrcoef calculates correlation between rows
    b_matrix = np.corrcoef(yield_matrix.T)

    print("\n" + "="*60)
    print("YOUR TRUE CROSS-FIDELITY CORRELATION MATRIX (B)")
    print("="*60)
    print(np.array2string(b_matrix, formatter={'float_kind': lambda x: f"{x:.4f}"}))
    print("="*60)
    print("\nTake this printout matrix and replace the placeholder 'B' array in your optimization script.")
    print("Note: Don't forget that before you submit the final project, you will need to rename your optimization script to 'main.py' and place it in a 'Beat-The-Felix' folder![cite: 1]")
    print("="*60)
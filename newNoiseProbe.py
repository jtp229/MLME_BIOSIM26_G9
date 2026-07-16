import time
import requests
import numpy as np
import basic_client


USER = "group09"
PASSWORD = "3v13c-n3qcr-y3qj9"
BASE_URL = "https://mlme26biosim.org"



if __name__ == "__main__":
    client = basic_client.BioreactorClient(BASE_URL)
    client.login(USER, PASSWORD)

    # Baseline recipe sitting in the middle of operating ranges
    BASELINE_RECIPE = {"T": 40.0, "pH": 6.25, "F1": 1.0, "F2": 1.0, "F3": 1.0}
    

    REPLICATE_COUNTS = {
        "micro": 50,
        "bench": 50,
        "pilot": 50
    }
    
    noise_results = {}
    raw_data_log = {}

    print("=" * 60)
    print("STARTING HIGH-PRECISION NOISE PROFILE (10 REPLICATES PER SCALE)")
    print("=" * 60)

    for scale, count in REPLICATE_COUNTS.items():
        print(f"\nProfiling scale: [{scale.upper()}]...")
        yields = []
        
        for i in range(count):
            print(f"  -> Executing run {i+1}/{count}...", end="", flush=True)
            start_time = time.time()
            
            res = client.run(scale, **BASELINE_RECIPE)
            y = float(res["Y"])
            yields.append(y)
            
            print(f" Completed! Y = {y:.4f} (took {time.time() - start_time:.1f}s)", flush=True)
            time.sleep(0.5)  # pacing interval for the server
        
        raw_data_log[scale] = yields
        # Delta degrees of freedom (ddof=1) ensures an unbiased sample variance estimate
        variance = float(np.var(yields, ddof=1))
        noise_results[scale] = variance
        
        print(f"\nFinished [{scale.upper()}] Profiling:")
        print(f"  Raw Yields: {[round(x, 4) for x in yields]}")
        print(f"  Calculated Sample Variance (Noise): {variance:.6f}")
        print("-" * 60)

    print("\n" + "=" * 60)
    print("PROFILING COMPLETE — COPY THIS INTO YOUR BO CONFIGURATION")
    print("=" * 60)
    print("SCALE_NOISE = {")
    print(f"    0: {noise_results['micro']:.6f},   # micro (calculated from {REPLICATE_COUNTS['micro']} runs)")
    print(f"    1: {noise_results['bench']:.6f},   # bench (calculated from {REPLICATE_COUNTS['bench']} runs)")
    print(f"    2: {noise_results['pilot']:.6f},   # pilot (calculated from {REPLICATE_COUNTS['pilot']} runs)")
    print("}")
    print("=" * 60)

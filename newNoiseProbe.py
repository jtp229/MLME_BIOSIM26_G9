import time
import requests
import numpy as np


USER = "group09"
PASSWORD = "3v13c-n3qcr-y3qj9"
BASE_URL = "https://mlme26biosim.org"

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
        last_err: Exception | None = None
        for attempt in range(8):
            try:
                r = self.s.post(
                    f"{self.base}/api/run", json=payload,
                    headers={"X-CSRF-Token": self._csrf()},
                    timeout=60,
                )
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                last_err = e
                wait = 2.5 ** attempt
                print(f"  [network error: {type(e).__name__}, sleeping {wait:.1f}s]", flush=True)
                time.sleep(wait)
                continue
            if r.status_code == 429:
                wait = 2.5 ** attempt
                print(f"  [rate-limited, sleeping {wait:.1f}s]", flush=True)
                time.sleep(wait)
                continue
            if r.status_code == 402:
                raise RuntimeError(f"group budget exhausted: {r.json().get('detail')}")
            if r.status_code >= 500:
                wait = 2.5 ** attempt
                print(f"  [server {r.status_code}, sleeping {wait:.1f}s]", flush=True)
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        raise RuntimeError(f"too many failed attempts (last error: {last_err!r})")


if __name__ == "__main__":
    client = BioreactorClient(BASE_URL)
    client.login(USER, PASSWORD)

    # A neutral baseline recipe sitting safely in the middle of your operating ranges
    BASELINE_RECIPE = {"T": 40.0, "pH": 6.25, "F1": 1.0, "F2": 1.0, "F3": 1.0}
    
    # 10 replicates per scale gives us 9 degrees of freedom each, 
    # which stabilizes the sample variance calculation significantly.
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
            time.sleep(0.5)  # Polite pacing interval for the server
        
        raw_data_log[scale] = yields
        # Delta degrees of freedom (ddof=1) ensures an unbiased sample variance estimate
        variance = float(np.var(yields, ddof=1))
        noise_results[scale] = variance
        
        print(f"\nFinished [{scale.upper()}] Profiling:")
        print(f"  Raw Yields: {[round(x, 4) for x in yields]}")
        print(f"  Calculated Sample Variance (Noise): {variance:.6f}")
        print("-" * 60)

    # ── Final Output Formatting ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PROFILING COMPLETE — COPY THIS INTO YOUR BO CONFIGURATION")
    print("=" * 60)
    print("SCALE_NOISE = {")
    print(f"    0: {noise_results['micro']:.6f},   # micro (calculated from {REPLICATE_COUNTS['micro']} runs)")
    print(f"    1: {noise_results['bench']:.6f},   # bench (calculated from {REPLICATE_COUNTS['bench']} runs)")
    print(f"    2: {noise_results['pilot']:.6f},   # pilot (calculated from {REPLICATE_COUNTS['pilot']} runs)")
    print("}")
    print("=" * 60)
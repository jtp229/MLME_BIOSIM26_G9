"""
Task 2: Quantify per-scale noise and bias.

Runs replicate measurements at 2 probe recipes, across all 3 scales,
and saves everything to a CSV for analysis/plotting.

Budget: 2 recipes x (10 micro x 10 EUR + 2 bench x 500 EUR + 1 pilot x 2000 EUR)
      = 2 x (100 + 1000 + 2000) = 6200 EUR total
"""
import csv
import time
import basic_client

client = basic_client.BioreactorClient(basic_client.BASE_URL)
client.login(basic_client.USER, basic_client.PASSWORD)

# --- Probe recipes ---
probe_recipes = {
    "baseline": {"T": 30.0, "pH": 6.5, "F1": 1.0, "F2": 1.0, "F3": 1.0},
    "hotspot":  {"T": 35.5, "pH": 6.5, "F1": 1.0, "F2": 1.0, "F3": 1.0},
}

# --- Replicate counts per scale ---
reps = {
    "micro": 10,
    "bench": 2,
    "pilot": 1,
}

rows = []

for recipe_name, recipe in probe_recipes.items():
    for scale, n_reps in reps.items():
        print(f"--- {recipe_name} @ {scale} ({n_reps} reps) ---")
        for i in range(n_reps):
            result = client.run(scale, **recipe)
            row = {
                "recipe_name": recipe_name,
                "scale": scale,
                "rep": i,
                **recipe,
                "Y": result["Y"],
                "cost_eur": result["cost_eur"],
                "total_cost_eur": result["total_cost_eur"],
                "run_id": result["id"],
            }
            rows.append(row)
            print(f"  rep {i}: Y={result['Y']:.4f}, "
                  f"cost={result['cost_eur']}, "
                  f"total_spent={result['total_cost_eur']}")
            time.sleep(0.3)

# --- Save to CSV ---
csv_filename = "noise_bias_data.csv"
fieldnames = ["recipe_name", "scale", "rep", "T", "pH", "F1", "F2", "F3",
              "Y", "cost_eur", "total_cost_eur", "run_id"]

with open(csv_filename, mode="w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"\nDone. Saved {len(rows)} rows to {csv_filename}")
print(f"Final total_cost_eur reported by server: {rows[-1]['total_cost_eur']}")
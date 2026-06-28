# test_bo.py
import numpy as np
from BO import (
    MultiFidelityKernel, expected_improvement,
    select_next_experiment, SCALE_COSTS, RECIPE_MIN, RECIPE_MAX,
    SCALE_NOISE, B
)
from sklearn.gaussian_process import GaussianProcessRegressor

# ── Test 1: B matrix is valid ─────────────────────────────────────────
def test_B_matrix():
    assert B.shape == (3, 3), "B must be 3x3"
    assert np.allclose(B, B.T), "B must be symmetric"
    assert np.all(np.diag(B) == 1.0), "diagonal must be 1.0"
    eigenvalues = np.linalg.eigvalsh(B)
    assert np.all(eigenvalues > 0), "B must be positive definite"
    print("✓ B matrix valid")

# ── Test 2: EI returns 0 when std is near zero ───────────────────────
def test_ei_zero_std():
    ei = expected_improvement(mean=5.0, std=1e-10, best_f=3.0)
    assert ei == 0.0, f"Expected 0.0, got {ei}"
    print("✓ EI handles zero std")

# ── Test 3: EI is higher for higher mean ─────────────────────────────
def test_ei_ordering():
    ei_high = expected_improvement(mean=10.0, std=1.0, best_f=5.0)
    ei_low  = expected_improvement(mean=5.5,  std=1.0, best_f=5.0)
    assert ei_high > ei_low, "Higher mean should give higher EI"
    print("✓ EI ordering correct")

# ── Test 4: EI is higher for higher std (exploration) ────────────────
def test_ei_exploration():
    ei_certain    = expected_improvement(mean=5.0, std=0.1, best_f=5.0)
    ei_uncertain  = expected_improvement(mean=5.0, std=2.0, best_f=5.0)
    assert ei_uncertain > ei_certain, "Higher std should give higher EI"
    print("✓ EI exploration term correct")

# ── Test 5: select_next_experiment doesn't always pick micro ─────────
def test_scale_selection_not_always_micro():
    # Build a GP where bench region is clearly better
    # Put high Y observations at bench scale, low at micro
    rng = np.random.default_rng(42)
    recipes = np.random.uniform(RECIPE_MIN, RECIPE_MAX, (20, 5))

    X_train = np.vstack([
        np.column_stack([recipes[:10], np.zeros(10)]),   # micro, low Y
        np.column_stack([recipes[10:], np.ones(10)]),    # bench, high Y
    ])
    Y_train = np.concatenate([
        np.full(10, 0.15),   # micro Y ~ 0.15
        np.full(10, 8.0),    # bench Y ~ 8.0
    ])

    kernel = MultiFidelityKernel(length_scale=5.0)
    alpha  = np.array([SCALE_NOISE[int(s)] for s in X_train[:, 5]])
    gp     = GaussianProcessRegressor(kernel=kernel, alpha=alpha,
                                      n_restarts_optimizer=0)
    gp.fit(X_train, Y_train)

    scales_chosen = []
    for _ in range(10):
        _, scale = select_next_experiment(gp, X_train, Y_train, n_candidates=500)
        scales_chosen.append(scale)

    assert not all(s == 0 for s in scales_chosen), \
        f"Algorithm always chose micro — Bug 1 may still be present. Scales: {scales_chosen}"
    print(f"✓ Scale selection varied: {dict(zip(*np.unique(scales_chosen, return_counts=True)))}")

# ── Test 6: kernel produces valid covariance matrix ──────────────────
def test_kernel_positive_definite():
    kernel  = MultiFidelityKernel(length_scale=5.0)
    X_test  = np.column_stack([
        np.random.uniform(RECIPE_MIN, RECIPE_MAX, (10, 5)),
        np.random.randint(0, 3, 10).astype(float),
    ])
    K = kernel(X_test)
    assert K.shape == (10, 10), "Kernel matrix wrong shape"
    eigenvalues = np.linalg.eigvalsh(K)
    assert np.all(eigenvalues > -1e-6), "Kernel matrix not positive semi-definite"
    print("✓ Kernel produces valid covariance matrix")

# ── Test 7: GP can fit and predict without crashing ──────────────────
def test_gp_fit_predict():
    X = np.column_stack([
        np.random.uniform(RECIPE_MIN, RECIPE_MAX, (15, 5)),
        np.random.randint(0, 3, 15).astype(float),
    ])
    Y = np.random.uniform(0.1, 10.0, 15)
    alpha = np.array([SCALE_NOISE[int(s)] for s in X[:, 5]])

    kernel = MultiFidelityKernel(length_scale=5.0)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha,
                                   n_restarts_optimizer=0)
    gp.fit(X, Y)

    X_pred = np.column_stack([
        np.random.uniform(RECIPE_MIN, RECIPE_MAX, (5, 5)),
        np.full(5, 2.0),
    ])
    mu, sigma = gp.predict(X_pred, return_std=True)
    assert mu.shape == (5,), "Wrong prediction shape"
    assert np.all(sigma >= 0), "Negative std in GP prediction"
    print("✓ GP fits and predicts correctly")

if __name__ == "__main__":
    print("=== Running offline unit tests ===\n")
    test_B_matrix()
    test_ei_zero_std()
    test_ei_ordering()
    test_ei_exploration()
    test_kernel_positive_definite()
    test_gp_fit_predict()
    test_scale_selection_not_always_micro()
    print("\n✓ All tests passed — safe to proceed to mock API layer")
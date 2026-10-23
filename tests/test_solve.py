"""The solver is where a silent mistake corrupts every result, so each closed form is compared
with an independent solve of the KKT system on vec(ΔW)."""
import numpy as np
import torch
import pytest

import weightir as wir

SITE = "layer"


def vec(M):
    return M.reshape(-1, order="F")


def unvec(x, m, n):
    return x.reshape(m, n, order="F")


def kkt(W, H, G, A, B):
    """min tr(ΔW H ΔWᵀ) − 2 tr(ΔW Gᵀ) s.t. ΔW A = B, solved on vec(ΔW) without the closed form."""
    m, n = W.shape
    I = np.eye(m)
    Q = np.kron(H, I)
    C = np.kron(A.T, I)
    k = C.shape[0]
    lhs = np.block([[2 * Q, C.T], [C, np.zeros((k, k))]])
    rhs = np.concatenate([2 * vec(G), vec(B)])
    return unvec(np.linalg.lstsq(lhs, rhs, rcond=None)[0][: m * n], m, n)


def problem(rng, m=5, n=12, p=3, s=3):
    W = rng.standard_normal((m, n))
    K = rng.standard_normal((n, p))
    V = rng.standard_normal((m, p))
    U = np.linalg.qr(rng.standard_normal((n, s)))[0]
    X = rng.standard_normal((n, 40))
    C = X @ X.T / 40
    return W, K, V, U, C


def keys(K):
    return wir.Keys(K, SITE)


def values(V):
    return wir.Values(V, SITE)


@pytest.mark.parametrize("seed", range(5))
def test_exact_frobenius_matches_kkt(seed):
    rng = np.random.default_rng(seed)
    W, K, V, U, _ = problem(rng)
    e = wir.edit(SITE, map=wir.Exact(keys(K), values(V)), preserve=wir.Exact(wir.Subspace(U, SITE)))
    D, _ = wir.solve(e, W)
    ref = kkt(W, np.eye(W.shape[1]), np.zeros_like(W), np.hstack([K, U]),
              np.hstack([V - W @ K, np.zeros((W.shape[0], U.shape[1]))]))
    assert np.allclose(D, ref, atol=1e-10)
    assert np.allclose((W + D) @ K, V, atol=1e-10)
    assert np.allclose(D @ U, 0, atol=1e-10)


@pytest.mark.parametrize("seed", range(5))
def test_exact_metric_matches_kkt_and_rome(seed):
    rng = np.random.default_rng(seed)
    W, K, V, U, C = problem(rng)
    cov = wir.Covariance(C, SITE)
    e = wir.edit(SITE, map=wir.Exact(keys(K), values(V)), preserve=wir.Exact(wir.Subspace(U, SITE)),
                 minimize=cov)
    D, _ = wir.solve(e, W)
    ref = kkt(W, C, np.zeros_like(W), np.hstack([K, U]),
              np.hstack([V - W @ K, np.zeros((W.shape[0], U.shape[1]))]))
    assert np.allclose(D, ref, atol=1e-9)
    # ROME: one key, no preserved subspace
    k, v = K[:, :1], V[:, :1]
    D1, _ = wir.solve(wir.edit(SITE, map=wir.Exact(keys(k), values(v)), minimize=cov), W)
    Cik = np.linalg.solve(C, k)
    rome = (v - W @ k) @ Cik.T / (k.T @ Cik)
    assert np.allclose(D1, rome, atol=1e-10)


@pytest.mark.parametrize("seed", range(5))
def test_soft_with_exact_preserve_matches_kkt(seed):
    """MEMIT/AlphaEdit form: soft maps, soft covariance penalty, exact preserve, ridge."""
    rng = np.random.default_rng(seed)
    W, K, V, U, C = problem(rng, p=4)
    e = wir.edit(SITE, map=wir.Soft(keys(K), values(V), weight=2.0),
                 preserve=[wir.Exact(wir.Subspace(U, SITE)), wir.Soft(wir.Covariance(C, SITE), weight=0.5)],
                 ridge=0.3)
    D, _ = wir.solve(e, W)
    n = W.shape[1]
    H = np.eye(n) + 2.0 * K @ K.T + 0.5 * C + 0.3 * np.eye(n)
    G = 2.0 * (V - W @ K) @ K.T
    ref = kkt(W, H, G, U, np.zeros((W.shape[0], U.shape[1])))
    assert np.allclose(D, ref, atol=1e-9)


def test_memit_closed_form():
    rng = np.random.default_rng(7)
    W, K, V, _, C = problem(rng, p=6)
    lam = 3.0
    D, _ = wir.solve(wir.edit(SITE, map=wir.Soft(keys(K), values(V)), minimize=wir.Covariance(lam * C, SITE)), W)
    memit = (V - W @ K) @ K.T @ np.linalg.inv(lam * C + K @ K.T)
    assert np.allclose(D, memit, atol=1e-10)


def test_mixed_exact_and_soft_matches_kkt():
    rng = np.random.default_rng(3)
    W, K, V, U, C = problem(rng, p=2)
    K2, V2 = rng.standard_normal((W.shape[1], 3)), rng.standard_normal((W.shape[0], 3))
    e = wir.edit(SITE, map=[wir.Exact(keys(K), values(V)), wir.Soft(keys(K2), values(V2), weight=1.5)],
                 preserve=wir.Exact(wir.Subspace(U, SITE)), minimize=wir.Covariance(C, SITE))
    D, _ = wir.solve(e, W)
    H = C + 1.5 * K2 @ K2.T
    G = 1.5 * (V2 - W @ K2) @ K2.T
    ref = kkt(W, H, G, np.hstack([K, U]), np.hstack([V - W @ K, np.zeros((W.shape[0], U.shape[1]))]))
    assert np.allclose(D, ref, atol=1e-9)


def test_conflict_is_detected_with_witness():
    rng = np.random.default_rng(0)
    W, K, V, U, _ = problem(rng)
    k3 = K[:, 0] + K[:, 1] + U @ rng.standard_normal(U.shape[1])     # dependent after removing span(U)
    Kd = np.column_stack([K[:, 0], K[:, 1], k3])
    v3 = W @ k3 + (V[:, 0] - W @ K[:, 0]) + (V[:, 1] - W @ K[:, 1])   # consistent target
    ok = wir.edit(SITE, map=wir.Exact(keys(Kd), values(np.column_stack([V[:, 0], V[:, 1], v3]))),
                  preserve=wir.Exact(wir.Subspace(U, SITE)))
    D, _ = wir.solve(ok, W)
    assert np.allclose((W + D) @ Kd, ok.maps[0].V.data, atol=1e-9)
    bad = wir.edit(SITE, map=wir.Exact(keys(Kd), values(np.column_stack([V[:, 0], V[:, 1], v3 + 0.5]))),
                   preserve=wir.Exact(wir.Subspace(U, SITE)))
    with pytest.raises(wir.Infeasible) as info:
        wir.solve(bad, W)
    assert info.value.kind == "conflict"
    assert set(info.value.details["witness"]) == {0, 1, 2}


def test_key_inside_preserved_subspace_with_new_value_conflicts():
    rng = np.random.default_rng(1)
    W, _, V, U, _ = problem(rng)
    k = U[:, :1] * 2.0
    with pytest.raises(wir.Infeasible) as info:
        wir.solve(wir.edit(SITE, map=wir.Exact(keys(k), values(V[:, :1])), preserve=wir.Exact(wir.Subspace(U, SITE))), W)
    assert info.value.kind == "conflict"


def test_near_dependent_keys_are_unresolved():
    rng = np.random.default_rng(2)
    W, K, V, _, _ = problem(rng)
    k3 = K[:, 0] + 1e-9 * rng.standard_normal(W.shape[1])          # inside the (1e-10, 1e-7] band
    Kd = np.column_stack([K[:, 0], k3])
    with pytest.raises(wir.Infeasible) as info:
        wir.solve(wir.edit(SITE, map=wir.Exact(keys(Kd), values(V[:, :2]))), W)
    assert info.value.kind == "unresolved"


def test_rank_budget():
    rng = np.random.default_rng(4)
    W, K, _, _, C = problem(rng, p=4)
    R1 = np.outer(rng.standard_normal(W.shape[0]), rng.standard_normal(4))   # rank-1 required change
    e = wir.edit(SITE, map=wir.Exact(keys(K), values(W @ K + R1)), require=wir.Rank(1))
    D, _ = wir.solve(e, W)
    assert np.linalg.matrix_rank(D, tol=1e-9) == 1
    V = rng.standard_normal((W.shape[0], 4))
    with pytest.raises(wir.Infeasible) as info:
        wir.solve(wir.edit(SITE, map=wir.Exact(keys(K), values(V)), require=wir.Rank(2)), W)
    assert info.value.kind == "rank" and info.value.details["required_rank"] == 4
    # soft problem: the truncated solution is the best rank-r solution in the objective's metric
    soft = wir.edit(SITE, map=wir.Soft(keys(K), values(V)), minimize=wir.Covariance(C, SITE), require=wir.Rank(2))
    Dr, info2 = wir.solve(soft, W)
    assert info2.get("rank_truncated") and np.linalg.matrix_rank(Dr, tol=1e-9) == 2

    def objective(D):
        return np.sum((D @ K - (V - W @ K)) ** 2) + np.trace(D @ C @ D.T)

    full, _ = wir.solve(wir.edit(SITE, map=wir.Soft(keys(K), values(V)), minimize=wir.Covariance(C, SITE)), W)
    Uf, sf, Vtf = np.linalg.svd(full)
    naive = (Uf[:, :2] * sf[:2]) @ Vtf[:2]
    assert objective(Dr) <= objective(naive) + 1e-12
    for _ in range(20):        # random rank-2 perturbations around the answer never do better
        P = rng.standard_normal((W.shape[0], 2)) @ rng.standard_normal((2, W.shape[1])) * 1e-3
        Ua, sa, Vta = np.linalg.svd(Dr + P)
        cand = (Ua[:, :2] * sa[:2]) @ Vta[:2]
        assert objective(Dr) <= objective(cand) + 1e-12


def test_norm_budget():
    rng = np.random.default_rng(5)
    W, K, V, U, C = problem(rng)
    base = wir.edit(SITE, map=wir.Exact(keys(K), values(V)), preserve=wir.Exact(wir.Subspace(U, SITE)))
    least = np.linalg.norm(wir.solve(base, W)[0])
    with pytest.raises(wir.Infeasible) as info:
        wir.solve(wir.edit(SITE, map=base.maps, preserve=base.preserve, require=wir.Norm(0.9 * least)), W)
    assert info.value.kind == "norm" and np.isclose(info.value.details["min_norm"], least)
    # metric objective whose solution is larger than the Frobenius budget: the budget wins
    e = wir.edit(SITE, map=base.maps, preserve=base.preserve, minimize=wir.Covariance(C, SITE),
                 require=wir.Norm(1.0001 * least))
    D, info2 = wir.solve(e, W)
    assert np.linalg.norm(D) <= 1.0001 * least * (1 + 1e-6)
    assert np.allclose((W + D) @ K, V, atol=1e-8) and np.allclose(D @ U, 0, atol=1e-8)
    unbudgeted = wir.solve(wir.edit(SITE, map=base.maps, preserve=base.preserve, minimize=wir.Covariance(C, SITE)), W)[0]
    if np.linalg.norm(unbudgeted) > 1.0001 * least:
        assert info2["norm_penalty"] > 0


def test_then_with_keep_previous_equals_joint():
    rng = np.random.default_rng(6)
    W0, _, _, U, C = problem(rng)
    KA, VA = rng.standard_normal((12, 2)), rng.standard_normal((5, 2))
    KB, VB = rng.standard_normal((12, 2)), rng.standard_normal((5, 2))

    class Toy:
        def __init__(self, W):
            self.W = W.copy()

        def weight(self, site):
            return self.W

        def apply(self, patch):
            self.W = self.W + patch.delta

    def make(K, V):
        return wir.edit(SITE, map=wir.Exact(keys(K), values(V)), preserve=wir.Exact(wir.Subspace(U, SITE)),
                        minimize=wir.Covariance(C, SITE))

    joint = wir.compile(wir.joint(make(KA, VA), make(KB, VB)), W0)
    for first, second in ((make(KA, VA), make(KB, VB)), (make(KB, VB), make(KA, VA))):
        toy = Toy(W0)
        wir.then(toy, first, lambda model, s=second: s)
        assert np.allclose(toy.W, W0 + joint.delta, atol=1e-9)
    # without keep_previous the order matters and the first edit is broken
    toy = Toy(W0)
    wir.then(toy, make(KA, VA), lambda model: make(KB, VB), keep_previous=False)
    assert not np.allclose(toy.W @ KA, VA, atol=1e-6)


def test_spec_rejects_wrong_sites_and_spaces():
    rng = np.random.default_rng(8)
    W, K, V, U, C = problem(rng)
    with pytest.raises(wir.SpecError):
        wir.edit(SITE, map=wir.Exact(wir.Keys(K, "other"), values(V)))
    with pytest.raises(wir.SpecError):
        wir.Exact(wir.Covariance(C, SITE))                    # exact preservation needs a subspace
    with pytest.raises(wir.SpecError):
        wir.edit(SITE, map=wir.Exact(keys(K), values(V)), preserve=wir.Exact(wir.Subspace(np.eye(5)[:, :2], SITE)))
    with pytest.raises(wir.SpecError):                         # keys/values swapped against the weight
        wir.solve(wir.edit(SITE, map=wir.Exact(wir.Keys(V, SITE), wir.Values(K, SITE))), W)


def test_check_reports_constraints_and_rounding():
    rng = np.random.default_rng(9)
    W, K, V, U, _ = problem(rng)
    sub = wir.subspace(U @ rng.standard_normal((3, 50)) + 1e-3 * rng.standard_normal((12, 50)), SITE, energy=0.99)
    e = wir.edit(SITE, map=wir.Exact(keys(K), values(V)), preserve=wir.Exact(sub))
    p = wir.compile(e, W, dtype="bfloat16")
    assert p.check["ok"] and p.check["map_residual"] < 1e-9 and p.check["preserve_residual"] < 1e-9
    assert 0 < p.check["preserve_data_change"] < 1e-2                 # approximate subspace: data moves a little
    assert p.check["preserve_change_bfloat16"] > p.check["preserve_residual"]   # rounding breaks exactness


def test_bfloat16_rounding_matches_torch():
    x = np.random.default_rng(10).standard_normal(10_000) * 10.0 ** np.random.default_rng(11).integers(-8, 8, 10_000)
    ours = wir.round_to(x, "bfloat16")
    ref = torch.tensor(x, dtype=torch.float64).to(torch.bfloat16).to(torch.float64).numpy()
    assert np.array_equal(ours, ref)

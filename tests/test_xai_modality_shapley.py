"""Unit tests for run_xai_modality_shapley.py.

Zero third-party test dependencies: run with

    python tests/test_xai_modality_shapley.py

or let pytest collect it later (the functions are plain asserts).

The game-theoretic quantities are checked against an analytically solvable
multi-linear game, so a bug in the coalition weights cannot hide:

    v(S) = sum_{i in S} a_i + sum_{i<j in S} b_ij
    => phi_i     = a_i + 0.5 * sum_{j != i} b_ij
    => I_ij      = b_ij                      (Grabisch-Roubens, pair)
    => F_ij      = b_ij                      (2x2 factorial)
    => m({i})    = a_i,  m({i,j}) = b_ij,    m(|S|>=3) = 0
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import find_cpg_candidate_positions  # noqa: E402
from run_xai_modality_shapley import (  # noqa: E402
    Player,
    factorial_interaction,
    harsanyi_dividends,
    pairwise_shapley_interaction,
    permuted_track,
    resolve_players,
    scored_positions,
    sequence_absence,
    shapley_values,
    subset_name,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def pairwise_game(n: int, a: np.ndarray, b: np.ndarray, dtype=torch.float64):
    """b is an (n, n) symmetric matrix with zero diagonal."""
    values = {}
    for mask in range(1 << n):
        total = sum(a[i] for i in range(n) if (mask >> i) & 1)
        for i in range(n):
            if not (mask >> i) & 1:
                continue
            for j in range(i + 1, n):
                if (mask >> j) & 1:
                    total += b[i, j]
        values[mask] = torch.tensor([total], dtype=dtype)
    return values


def random_game(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    return {mask: torch.tensor(rng.normal(size=3), dtype=torch.float64)
            for mask in range(1 << n)}


def one_hot(ids: np.ndarray) -> torch.Tensor:
    return torch.nn.functional.one_hot(torch.from_numpy(ids).long(), 4).float().unsqueeze(0)


def expect_exit(fn, label: str):
    try:
        fn()
    except SystemExit:
        return
    raise AssertionError(f"{label}: expected SystemExit, nothing was raised")


# ---------------------------------------------------------------------------
# Shapley values / interaction indices / dividends
# ---------------------------------------------------------------------------

def test_pairwise_game_matches_analytic_solution():
    for n in (2, 3, 4, 5):
        rng = np.random.default_rng(n)
        a = rng.normal(size=n)
        b = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                b[i, j] = b[j, i] = rng.normal()
        values = pairwise_game(n, a, b)

        phi = shapley_values(values, n)
        for i in range(n):
            expected = a[i] + 0.5 * sum(b[i, j] for j in range(n) if j != i)
            assert torch.allclose(phi[i], torch.tensor([expected], dtype=torch.float64),
                                  atol=1e-12), f"n={n} phi_{i} mismatch"

        inter = pairwise_shapley_interaction(values, n)
        fact = factorial_interaction(values, n)
        div = harsanyi_dividends(values, n)
        for i in range(n):
            for j in range(i + 1, n):
                target = torch.tensor([b[i, j]], dtype=torch.float64)
                assert torch.allclose(inter[(i, j)], target, atol=1e-12), f"I_{i}{j}"
                assert torch.allclose(fact[(i, j)], target, atol=1e-12), f"F_{i}{j}"
                assert torch.allclose(div[(1 << i) | (1 << j)], target, atol=1e-12)
            assert torch.allclose(div[1 << i],
                                  torch.tensor([a[i]], dtype=torch.float64), atol=1e-12)
        for mask in range(1 << n):
            if mask.bit_count() >= 3:
                assert torch.allclose(div[mask], torch.zeros(1, dtype=torch.float64),
                                      atol=1e-12), f"n={n} m({mask}) should vanish"


def test_shapley_efficiency_and_dividend_identity():
    for n in (4, 5):
        values = random_game(n, seed=100 + n)
        full = (1 << n) - 1
        total = values[full] - values[0]

        phi = shapley_values(values, n)
        assert torch.allclose(sum(phi), total, atol=1e-10), f"n={n} efficiency"

        div = harsanyi_dividends(values, n)
        assert torch.allclose(sum(div[m] for m in range(1, 1 << n)), total,
                              atol=1e-10), f"n={n} dividend identity"


def test_harsanyi_dividends_reconstruct_every_coalition():
    n = 5
    values = random_game(n, seed=7)
    div = harsanyi_dividends(values, n)
    for subset in range(1 << n):
        reconstructed = sum(div[inner] for inner in range(1 << n)
                            if (inner & subset) == inner)
        assert torch.allclose(reconstructed, values[subset], atol=1e-10), \
            f"subset {subset}"


def test_additive_game_has_no_interaction():
    n = 4
    a = np.array([1.0, -2.0, 0.5, 3.0])
    values = pairwise_game(n, a, np.zeros((n, n)))
    phi = shapley_values(values, n)
    zero = torch.zeros(1, dtype=torch.float64)
    for i in range(n):
        assert torch.allclose(phi[i], torch.tensor([a[i]], dtype=torch.float64), atol=1e-12)
    for _key, tensor in pairwise_shapley_interaction(values, n).items():
        assert torch.allclose(tensor, zero, atol=1e-12)
    for _key, tensor in factorial_interaction(values, n).items():
        assert torch.allclose(tensor, zero, atol=1e-12)


def test_vectorised_over_positions():
    """The same code path must work for (B, L) tensors, not just scalars."""
    n = 4
    rng = np.random.default_rng(3)
    a = rng.normal(size=n)
    b = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            b[i, j] = b[j, i] = rng.normal()
    base = pairwise_game(n, a, b)
    shape = (2, 5)
    values = {mask: tensor.expand(*shape).clone() for mask, tensor in base.items()}

    phi = shapley_values(values, n)
    assert phi[0].shape == shape
    for i in range(n):
        expected = a[i] + 0.5 * sum(b[i, j] for j in range(n) if j != i)
        assert torch.allclose(phi[i], torch.full(shape, expected, dtype=torch.float64),
                              atol=1e-12)

    inter = pairwise_shapley_interaction(values, n)
    assert torch.allclose(inter[(0, 1)],
                          torch.full(shape, b[0, 1], dtype=torch.float64), atol=1e-12)


def test_subset_name():
    players = [Player("5mc", 0, "track", "5mc"),
               Player("seq", None, "seq", "seq"),
               Player("atac", 2, "track", "atac")]
    assert subset_name(players, 0) == ""
    assert subset_name(players, 0b001) == "5mc"
    assert subset_name(players, 0b101) == "5mc+atac"
    assert subset_name(players, 0b111) == "5mc+seq+atac"


# ---------------------------------------------------------------------------
# Absence operators
# ---------------------------------------------------------------------------

def test_sequence_absence_keeps_c_and_g():
    seq = "ACGTTCGAACGTTACGCG"
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    ids = np.array([mapping[c] for c in seq], dtype=np.int64)
    out = sequence_absence(one_hot(ids), torch.from_numpy(ids).unsqueeze(0),
                           np.random.default_rng(0), mode="at")
    out_ids = out.argmax(-1)[0].numpy()
    keep = np.isin(ids, (1, 2))
    assert np.array_equal(out_ids[keep], ids[keep]), "C/G must not move"
    assert np.array_equal(np.sort(out_ids), np.sort(ids)), "composition must be preserved"
    before = find_cpg_candidate_positions(torch.from_numpy(ids).unsqueeze(0))[0].numpy()
    after = find_cpg_candidate_positions(torch.from_numpy(out_ids).unsqueeze(0))[0].numpy()
    assert np.array_equal(before, after), "CpG positions must be unchanged"


def test_permuted_track_preserves_support_and_marginal():
    x = torch.zeros(1, 12, 1)
    x[0, [1, 3, 5, 7, 9], 0] = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
    out = permuted_track(x, np.random.default_rng(1))
    assert torch.equal(out != 0, x != 0), "support must be identical"
    assert torch.allclose(torch.sort(out.flatten())[0], torch.sort(x.flatten())[0]), \
        "value marginal must be identical"


def test_permuted_track_single_value_is_noop():
    x = torch.zeros(1, 8, 1)
    x[0, 4, 0] = 0.7
    assert torch.equal(permuted_track(x, np.random.default_rng(2)), x)


# ---------------------------------------------------------------------------
# Position selection
# ---------------------------------------------------------------------------

def test_scored_positions_forward_cpg_only():
    # "C G G C C G A" -> forward CpG C at 0 and 4; G at 1 and 5 belong to the
    # same dinucleotides but carry no value in the CG bedGraph.
    ids = torch.tensor([[1, 2, 2, 1, 1, 2, 0]], dtype=torch.long)
    mask = torch.ones(1, 7, 1)
    assert scored_positions(ids, mask, "cpg_forward")[0].tolist() == \
        [True, False, False, False, True, False, False]
    assert scored_positions(ids, mask, "cpg")[0].tolist() == \
        [True, True, False, False, True, True, False]
    # an unmasked position is excluded even when it is a CpG
    mask[0, 4, 0] = 0.0
    assert scored_positions(ids, mask, "cpg_forward")[0].tolist() == \
        [True, False, False, False, False, False, False]


# ---------------------------------------------------------------------------
# Player resolution
# ---------------------------------------------------------------------------

def test_resolve_players_default_five():
    players = resolve_players(["seq", "other", "atac", "rna", "self"],
                              "5hmc", ["5mc", "5hmc"], ["atac", "rna"],
                              [[1, 2, 3], [0, 2, 3]])
    assert [p.name for p in players] == ["seq", "5mc", "atac", "rna", "5hmc"]
    assert len(players) == 5


def test_resolve_players_rejects_duplicate_slot():
    expect_exit(
        lambda: resolve_players(["seq", "5mc", "atac", "rna", "self"],
                                "5mc", ["5mc", "5hmc"], ["atac", "rna"],
                                [[1, 2, 3], [0, 2, 3]]),
        "duplicate 5mc slot",
    )


def test_resolve_players_rejects_non_context_track():
    # context_mode="shared": 5mc is not visible to the 5hmc head
    expect_exit(
        lambda: resolve_players(["seq", "5mc", "self"], "5hmc",
                                ["5mc", "5hmc"], ["atac", "rna"], [[2, 3], [2, 3]]),
        "5mc outside the 5hmc context",
    )


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

def main() -> int:
    tests = sorted(
        (name, obj) for name, obj in globals().items()
        if name.startswith("test_") and callable(obj)
    )
    failures = 0
    for name, fn in tests:
        try:
            fn()
        except Exception:
            failures += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
        else:
            print(f"ok    {name}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

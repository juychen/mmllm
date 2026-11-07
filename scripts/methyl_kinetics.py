#!/usr/bin/env python
"""C -> 5mC -> 5hmC -> C conversion kinetics for the mmllm multi-query model.

Three-state ODE (x = (c, m, h), x sums to 1):

    dc/dt = -k1*c + (k3+d)*h + d*m
    dm/dt =  k1*c - (k2+d)*m
    dh/dt =  k2*m - (k3+d)*h

    A = [[-k1, d, k3+d],
         [ k1, -(k2+d), 0],
         [ 0,  k2, -(k3+d)]]        # column sums are 0

    x(t) = expm(A*dt) x(t0)

Modes
-----
simulate : synthetic self-test. Verifies (a) the d=0 steady-state rate ratios
           (k1,k2,k3) ~ (1, c/m, c/h), (b) absolute-rate recovery when the slow
           mode is resolvable at the sampling interval, and (c) the honest
           negative result that a fast 5hmC turnover rate is NOT identifiable
           from 7-day-spaced samples. Writes a simplex phase portrait.
tracks   : real data. BED intervals + one or two time points of
           {5mC,5hmC} bedGraph (tabix-indexed). Emits per-interval
           composition, rate ratios, finite-difference velocity.
table    : same math on a pre-built per-position TSV.

Requires numpy + scipy; pysam only for `tracks`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

SQRT3_2 = math.sqrt(3.0) / 2.0


# --------------------------------------------------------------------------
# core math
# --------------------------------------------------------------------------
def build_A(k1, k2, k3, d=0.0):
    """Generator matrix of the three-state conversion cycle."""
    return np.array(
        [
            [-k1, d, k3 + d],
            [k1, -(k2 + d), 0.0],
            [0.0, k2, -(k3 + d)],
        ],
        dtype=float,
    )


def evolve(x0, k, dt, d=0.0):
    """Exact solution x(t0+dt) = expm(A*dt) x(t0)."""
    from scipy.linalg import expm

    return expm(build_A(k[0], k[1], k[2], d) * dt) @ np.asarray(x0, dtype=float)


def steady_state(k, d=0.0):
    """Normalised right null vector of A (the fixed point of the simplex)."""
    A = build_A(k[0], k[1], k[2], d)
    _, _, vt = np.linalg.svd(A)
    x = np.abs(vt[-1])
    return x / x.sum()


def rates_from_composition(x, d=0.0):
    """Recover (k1, k2, k3) up to the k1 gauge from one composition.

    d = 0 :  m/c = k1/k2  and  h/m = k2/k3
    d > 0 :  (k2+d)/k1 = c/m  and  (k3+d)/k2 = m/h   (d must be known)
    """
    c, m, h = np.clip(np.asarray(x, dtype=float), 1e-12, None)
    k1 = 1.0
    k2 = k1 * c / m - d
    k3 = k2 * m / h - d
    return np.array([k1, max(k2, 0.0), max(k3, 0.0)])


def qssa_h_over_m(k, d=0.0):
    """k3 >> k2 limit: h/m -> k2/(k3+d)."""
    return k[1] / (k[2] + d)


def velocity(x, k, d=0.0):
    """dx/dt = A x."""
    return build_A(k[0], k[1], k[2], d) @ np.asarray(x, dtype=float)


def flux(x, k, d=0.0):
    """Steady-state turnover flux J = k1*c = k2*m = k3*h (equal only at SS)."""
    c, m, h = np.clip(np.asarray(x, dtype=float), 0.0, None)
    return float(min(k[0] * c, (k[1] + d) * m, (k[2] + d) * h))


def project_simplex(v):
    """Clip to non-negative and renormalise to sum 1 (rows of a 2-D array too)."""
    v = np.clip(np.asarray(v, dtype=float), 0.0, None)
    s = v.sum(axis=-1, keepdims=True)
    s = np.where(s <= 0.0, 1.0, s)
    return v / s


def states_from_m5c_hm5c(m_pct, h_pct):
    """True-5mC / 5hmC percentages -> (c, m, h) on the simplex.

    The processed bedGraphs already follow the convention of
    `convert_bismark_cov_to_bedgraph.py`: the .m track is
    `WGBS - TAB` (clamped >= 0) and the .h track is TAB.  This matches
    `data.py` where c = 1 - m - h.  So:  c = 1 - m - h.
    """
    m = np.asarray(m_pct, dtype=float) / 100.0
    h = np.asarray(h_pct, dtype=float) / 100.0
    return project_simplex(np.stack([1.0 - m - h, m, h], axis=-1))


def fit_rates(X, ages, d=0.0, anchor=None, restarts=8, seed=0):
    """Least-squares fit of absolute (k1,k2,k3) from a composition time course.

    X      : (T, 3) observed compositions, in the same order as `ages` (days)
    anchor : if given, that row is held fixed as the initial condition.
             Default None = fit x0 jointly.  Anchoring on one noisy
             observation propagates its error into every rate estimate.
    Returns dict with k, x0, per-parameter std from the Jacobian, and cost.
    """
    from scipy.optimize import least_squares

    X = np.asarray(X, dtype=float)
    ages = np.asarray(ages, dtype=float)
    T = len(ages)

    if anchor is None:
        idx = list(range(T))
        t_off = ages - ages[0]
        x0_fixed = None
    else:
        x0_fixed = X[anchor]
        idx = [i for i in range(T) if i != anchor]
        t_off = ages[idx] - ages[anchor]

    def unpack(p):
        k = np.exp(p[:3])
        if x0_fixed is not None:
            return k, x0_fixed
        z = np.array([p[3], p[4], 0.0])
        z = z - z.max()
        x0 = np.exp(z)
        return k, x0 / x0.sum()

    def resid(p):
        k, x0 = unpack(p)
        return np.concatenate([evolve(x0, k, t, d) - X[i]
                               for i, t in zip(idx, t_off)])

    ratio = rates_from_composition(X[idx[-1]], d)
    scale = 0.05 / max(ratio[0], 1e-9)
    k0 = np.log(np.clip(ratio * scale, 1e-6, None))
    p0 = k0 if x0_fixed is not None else np.concatenate([k0, [0.0, 0.0]])

    rng = np.random.default_rng(seed)
    best = None
    for r in range(restarts):
        start = p0 if r == 0 else p0 + rng.normal(0.0, 1.0, size=p0.size)
        try:
            sol = least_squares(resid, np.clip(start, -12, 6),
                                bounds=(-16, 10), method="trf", max_nfev=6000)
        except Exception:
            continue
        if best is None or sol.cost < best.cost:
            best = sol
    if best is None:
        return {"k": np.full(3, np.nan), "k_std": np.full(3, np.nan),
                "cost": np.nan, "x0": np.full(3, np.nan)}

    k, x0 = unpack(best.x)
    J = best.jac
    dof = max(1, J.shape[0] - J.shape[1])
    try:
        cov = np.linalg.pinv(J.T @ J) * (2.0 * best.cost / dof)
        std = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    except Exception:
        std = np.full(p0.size, np.nan)
    return {"k": k, "k_std": k * std[:3], "cost": float(best.cost), "x0": x0}


# --------------------------------------------------------------------------
# simplex plotting
# --------------------------------------------------------------------------
def to_xy(x):
    """Barycentric -> 2-D.  c=(0,0), m=(1,0), h=(0.5, sqrt3/2)."""
    x = np.asarray(x, dtype=float)
    return x[..., 1] + 0.5 * x[..., 2], SQRT3_2 * x[..., 2]


def _draw_simplex(ax, title):
    ax.plot([0, 1, 0.5, 0], [0, 0, SQRT3_2, 0], color="0.25", lw=1.2)
    ax.text(-0.03, -0.05, "C", ha="right", va="top")
    ax.text(1.03, -0.05, "5mC", ha="left", va="top")
    ax.text(0.5, SQRT3_2 - 0.075, "5hmC", ha="center", va="top", color="0.25")
    ax.set_title(title, fontsize=10, pad=10)
    ax.set_aspect("equal")
    ax.axis("off")


def _arrow_field(ax, k, d=0.0, n=17, scale=0.075):
    g = np.linspace(0.0, 1.0, n)
    pts = []
    for c in g:
        for m in g:
            h = 1.0 - c - m
            if h < 0:
                continue
            pts.append((c, m, h))
    pts = np.array(pts)
    xy = np.stack(to_xy(pts), axis=-1)
    V = np.stack([velocity(p, k, d) for p in pts])
    u, v = to_xy(V)
    norm = np.hypot(u, v)
    keep = norm > 1e-12
    u = np.where(norm > 0, u / np.where(norm > 0, norm, 1) * scale, 0)
    v = np.where(norm > 0, v / np.where(norm > 0, norm, 1) * scale, 0)
    speed = norm / np.where(norm > 0, norm, 1)
    ax.quiver(
        xy[keep, 0], xy[keep, 1], u[keep], v[keep],
        color="0.55", angles="xy", scale_units="xy", scale=1.0,
        width=0.0025, headwidth=3.5, headlength=3.5,
    )


# --------------------------------------------------------------------------
# mode: simulate
# --------------------------------------------------------------------------
def cmd_simulate(args):
    rng = np.random.default_rng(args.seed)
    ages = np.array([0.0, 7.0, 56.0])
    conc = args.concentration

    print("=" * 72)
    print("CASE A  slow cycle, resolvable at 7 d spacing")
    print("=" * 72)
    kA = np.array([0.02, 0.05, 0.10])
    x0A = np.array([0.30, 0.55, 0.15])
    XA = np.stack([evolve(x0A, kA, t) for t in ages])
    XAn = np.stack([rng.dirichlet(np.clip(x * conc, 1e-6, None)) for x in XA])
    print(f"  true k        : {kA}  (1/day)")
    print(f"  SS composition: {np.round(steady_state(kA), 4)}")
    print(f"  true ratios   : (1, {kA[1]/kA[0]:.2f}, {kA[2]/kA[0]:.2f})")

    print("  -- composition-derived ratios (no fit, assumes steady state) --")
    for i, t in enumerate(ages):
        r = rates_from_composition(XAn[i])
        print(f"     t={t:5.1f}d  x={np.round(XAn[i],3)}  ratios=(1, {r[1]:.2f}, {r[2]:.2f})")

    fitA = fit_rates(XAn, ages, seed=args.seed)
    relA = np.abs(fitA["k"] - kA) / kA
    print(f"  -- integrated ODE fit --")
    print(f"     k_hat       : {np.round(fitA['k'], 5)}")
    print(f"     +- std      : {np.round(fitA['k_std'], 5)}")
    print(f"     rel. error  : {np.round(relA * 100, 1)} %")

    print()
    print("=" * 72)
    print("CASE B  fast 5hmC: the h transient is over before the first sample")
    print("=" * 72)
    k2_true = 0.5
    x0B = np.array([0.35, 0.55, 0.10])
    relaxed, ratios = {}, {}
    for k3 in (2.0, 8.0, 50.0):
        kB = np.array([0.05, k2_true, k3])
        ss = steady_state(kB)
        x_early = evolve(x0B, kB, 0.1)      # 2.4 h
        x7 = evolve(x0B, kB, 7.0)
        relaxed[k3] = abs(x7[2] - ss[2]) / max(abs(x0B[2] - ss[2]), 1e-12)
        ratios[k3] = ss[2] / ss[1]
        print(f"  k3={k3:5.1f}  x(0.1d)={np.round(x_early,4)}  x(7d)={np.round(x7,4)}"
              f"  SS={np.round(ss,4)}")
        print(f"          |x(7d)-SS| / |x0-SS| = {relaxed[k3]:.2e}"
              f"   (1/k3 = {1.0 / k3:.3f} d -> gone by P7)")
    print(f"  -> k3 is pinned only by the steady-state h/m level:")
    print(f"     k3=2 -> h/m={ratios[2.0]:.5f} ; k3=8 -> {ratios[8.0]:.5f} ;"
          f" k3=50 -> {ratios[50.0]:.5f}")
    print(f"     i.e. h/m = k2/k3, a SMALL number, so the estimate is a ratio of")
    print(f"     small counts. Relative sd under Poisson sampling ~ sqrt(1/n_h+1/n_m):")
    for cov in (3, 10, 30, 100):
        for hfrac in (0.02, 0.10):
            lam_h = cov * hfrac
            lam_m = cov * 0.30
            rsd = 100.0 * math.sqrt(1.0 / max(lam_h, 1e-9) + 1.0 / max(lam_m, 1e-9))
            print(f"       coverage {cov:3d}x, h={hfrac:.0%}:  sd(k3/k1) ~ {rsd:5.1f} %"
                  f"   (n_h={lam_h:.2f}, n_m={lam_m:.2f})")
    okB = bool(max(relaxed.values()) < 5e-2)

    okA = bool(relA.max() < 0.25)

    if not args.no_plot:
        _plot_phase(kA, x0A, ages, XAn, kA_fit=fitA["k"], path=args.out + ".png")
        print(f"\n  phase portrait -> {args.out}.png")

    summary = {
        "case_a": {
            "k_true": kA.tolist(),
            "k_hat": fitA["k"].tolist(),
            "k_std": fitA["k_std"].tolist(),
            "rel_err": relA.tolist(),
            "pass": okA,
        },
        "case_b": {
            "k2": k2_true,
            "k3_grid": [2.0, 8.0, 50.0],
            "h_over_m_at_ss": {str(k): float(ratios[k]) for k in ratios},
            "relaxation_by_7d": {str(k): float(relaxed[k]) for k in relaxed},
            "note": ("transient of h ends before P7 (1/k3 << 7 d); k3 is fixed only "
                     "by the small steady-state h/m ratio, so its relative error is "
                     "amplified by Poisson counting noise"),
            "pass": okB,
        },
    }
    with open(args.out + ".json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"  summary -> {args.out}.json")

    if args.check:
        if okA and okB:
            print("\nSELF-CHECK PASSED")
            return 0
        print("\nSELF-CHECK FAILED", file=sys.stderr)
        return 1
    return 0


def _plot_phase(kA, x0A, ages, XAn, kA_fit=None, path="phase.png"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.2))
    fast = np.array([0.05, 0.5, 8.0])

    for ax, k, x0, title in (
        (axes[0], kA, x0A, "slow cycle (resolvable): damped spiral"),
        (axes[1], fast, np.array([0.35, 0.55, 0.10]),
         "fast 5hmC (k3=8): overdamped, h pinned to m"),
    ):
        _draw_simplex(ax, title)
        _arrow_field(ax, k)
        ss = steady_state(k)
        ax.plot(*to_xy(ss), "k*", ms=12, label="steady state")

        ts = np.linspace(0.0, 60.0, 400)
        traj = np.stack([evolve(x0, k, t) for t in ts])
        xy = np.stack(to_xy(traj), axis=-1)
        ax.plot(xy[:, 0], xy[:, 1], color="tab:red", lw=1.4, label="trajectory")

        if np.allclose(k, kA):
            obs = XAn
            lbl = "observed (P0/P7/Adult)"
        else:
            obs = np.stack([evolve(x0, fast, t) for t in ages])
            lbl = "observed (P0/P7/Adult)"
        ax.plot(*to_xy(obs), "o", color="tab:blue", ms=6, label=lbl)

        if kA_fit is not None and np.allclose(k, kA):
            ss2 = steady_state(kA_fit)
            ax.plot(*to_xy(ss2), "x", color="tab:green", ms=10,
                    label="SS from fitted k")
        ax.legend(loc="upper right", fontsize=7, frameon=False)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------
# mode: tracks  (real bedGraph input)
# --------------------------------------------------------------------------
def _fetch_mean(tbx, chrom, start, end):
    vals = []
    try:
        for line in tbx.fetch(chrom, start, end):
            f = line.rstrip("\n").split("\t")
            if len(f) >= 4:
                vals.append(float(f[3]))
    except Exception:
        return float("nan")
    return float(np.mean(vals)) if vals else float("nan")


def _open_tabix(path):
    import pysam

    return pysam.TabixFile(path)


def cmd_tracks(args):
    import pysam  # noqa: F401  (fail early with a clear ImportError)

    tracks = {}
    for key in ("m5c_t0", "hm5c_t0", "m5c_t1", "hm5c_t1"):
        p = getattr(args, key)
        if p:
            tracks[key] = _open_tabix(p)
            print(f"  opened {key}: {p}  ({len(tracks[key].contigs)} contigs)")

    rows = []
    with open(args.bed) as fh:
        for line in fh:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            f = line.rstrip("\n").split("\t")
            chrom, start, end = f[0], int(f[1]), int(f[2])
            if args.strip_chr and chrom.startswith("chr"):
                chrom = chrom[3:]
            name = f[3] if len(f) > 3 else f"{chrom}:{start}-{end}"

            M0 = _fetch_mean(tracks["m5c_t0"], chrom, start, end)
            K0 = _fetch_mean(tracks["hm5c_t0"], chrom, start, end)
            if not np.isfinite(M0) or not np.isfinite(K0):
                continue
            x0 = states_from_m5c_hm5c(M0, K0)
            r0 = rates_from_composition(x0)

            row = {
                "chrom": chrom, "start": start, "end": end, "name": name,
                "m5c_0": M0, "hm5c_0": K0,
                "c0": x0[0], "m0": x0[1], "h0": x0[2],
                "k1_r0": 1.0, "k2_r0": r0[1], "k3_r0": r0[2],
                "h_over_m0": x0[2] / max(x0[1], 1e-9),
            }

            if "m5c_t1" in tracks and "hm5c_t1" in tracks:
                M1 = _fetch_mean(tracks["m5c_t1"], chrom, start, end)
                K1 = _fetch_mean(tracks["hm5c_t1"], chrom, start, end)
                if np.isfinite(M1) and np.isfinite(K1):
                    x1 = states_from_m5c_hm5c(M1, K1)
                    dt = float(args.dt)
                    v = (x1 - x0) / dt
                    r1 = rates_from_composition(x1)
                    row.update({
                        "m5c_1": M1, "hm5c_1": K1,
                        "c1": x1[0], "m1": x1[1], "h1": x1[2],
                        "dc_dt": v[0], "dm_dt": v[1], "dh_dt": v[2],
                        "k2_r1": r1[1], "k3_r1": r1[2],
                        "h_over_m1": x1[2] / max(x1[1], 1e-9),
                        # QSSA residual: |h - (k2/k3) m| relative to h
                        "qssa_resid": abs(x1[2] - r1[1] / max(r1[2], 1e-9) * x1[1])
                        / max(x1[2], 1e-9),
                    })
            rows.append(row)
            if len(rows) % max(1, args.progress) == 0:
                print(f"    {len(rows)} intervals ...")

    _write_rows(rows, args, kind="tracks")
    return 0


# --------------------------------------------------------------------------
# mode: table
# --------------------------------------------------------------------------
def cmd_table(args):
    rows = []
    with open(args.table) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        idx = {h: i for i, h in enumerate(header)}
        two_tp = "c1" in idx
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < len(header):
                continue
            c0, m0, h0 = (float(f[idx["c0"]]), float(f[idx["m0"]]),
                          float(f[idx["h0"]]))
            x0 = project_simplex([c0, m0, h0])
            r0 = rates_from_composition(x0)
            row = {
                "name": f[idx.get("name", 0)],
                "c0": x0[0], "m0": x0[1], "h0": x0[2],
                "k2_r0": r0[1], "k3_r0": r0[2],
                "h_over_m0": x0[2] / max(x0[1], 1e-9),
            }
            if two_tp:
                x1 = project_simplex([float(f[idx["c1"]]), float(f[idx["m1"]]),
                                      float(f[idx["h1"]])])
                v = (x1 - x0) / float(args.dt)
                r1 = rates_from_composition(x1)
                row.update({
                    "c1": x1[0], "m1": x1[1], "h1": x1[2],
                    "dc_dt": v[0], "dm_dt": v[1], "dh_dt": v[2],
                    "k2_r1": r1[1], "k3_r1": r1[2],
                    "h_over_m1": x1[2] / max(x1[1], 1e-9),
                })
            rows.append(row)
    _write_rows(rows, args, kind="table")
    return 0


def _write_rows(rows, args, kind):
    if not rows:
        print("no rows produced", file=sys.stderr)
        return
    keys = list(rows[0].keys())
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out + ".tsv", "w") as fh:
        fh.write("\t".join(keys) + "\n")
        for r in rows:
            fh.write("\t".join(
                f"{r[k]:.6g}" if isinstance(r[k], float) else str(r[k]) for k in keys
            ) + "\n")

    k2 = np.array([r["k2_r0"] for r in rows], dtype=float)
    k3 = np.array([r["k3_r0"] for r in rows], dtype=float)
    hom = np.array([r["h_over_m0"] for r in rows], dtype=float)
    print(f"\n{len(rows)} intervals -> {args.out}.tsv")
    print(f"  k2/k1  median {np.nanmedian(k2):8.3f}   IQR "
          f"[{np.nanpercentile(k2,25):.2f}, {np.nanpercentile(k2,75):.2f}]")
    print(f"  k3/k1  median {np.nanmedian(k3):8.3f}   IQR "
          f"[{np.nanpercentile(k3,25):.2f}, {np.nanpercentile(k3,75):.2f}]")
    print(f"  h/m    median {np.nanmedian(hom):8.4f}")
    if "h_over_m1" in rows[0]:
        hom1 = np.array([r["h_over_m1"] for r in rows if np.isfinite(r.get("h_over_m1", np.nan))])
        if hom1.size:
            print(f"  h/m at t1 median {np.nanmedian(hom1):8.4f}"
                  f"   (QSSA: should match t0 if k2/k3 is age-invariant)")
    summ = {
        "n": len(rows), "kind": kind,
        "k2_over_k1_median": float(np.nanmedian(k2)),
        "k3_over_k1_median": float(np.nanmedian(k3)),
        "h_over_m_median": float(np.nanmedian(hom)),
    }
    with open(args.out + ".json", "w") as fh:
        json.dump(summ, fh, indent=2)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    s = sub.add_parser("simulate", help="synthetic self-test + phase portrait")
    s.add_argument("--out", default="kinetics_sim")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--concentration", type=float, default=5000.0,
                   help="Dirichlet concentration; larger = less noise")
    s.add_argument("--no-plot", action="store_true")
    s.add_argument("--check", action="store_true", help="exit nonzero if the self-test fails")
    s.set_defaults(func=cmd_simulate)

    t = sub.add_parser("tracks", help="BED + bedGraph time points")
    t.add_argument("--bed", required=True)
    t.add_argument("--m5c-t0", required=True)
    t.add_argument("--hm5c-t0", required=True)
    t.add_argument("--m5c-t1", default=None)
    t.add_argument("--hm5c-t1", default=None)
    t.add_argument("--dt", type=float, default=None, help="days between t0 and t1")
    t.add_argument("--strip-chr", action="store_true")
    t.add_argument("--progress", type=int, default=2000)
    t.add_argument("--out", default="kinetics_tracks")
    t.set_defaults(func=cmd_tracks)

    b = sub.add_parser("table", help="per-position TSV")
    b.add_argument("--table", required=True)
    b.add_argument("--dt", type=float, default=None)
    b.add_argument("--out", default="kinetics_table")
    b.set_defaults(func=cmd_table)

    args = ap.parse_args(argv)
    if args.mode == "tracks" and args.m5c_t1 and args.dt is None:
        ap.error("--dt is required when t1 tracks are given")
    if args.mode == "table" and args.dt is None:
        args.dt = 1.0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

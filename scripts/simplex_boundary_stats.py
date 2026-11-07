"""Empirical check: how often would an unconstrained 2-head model violate m+h<=1?

Reads MC_AMY .m / .h bedGraphs in lockstep, pairs by coordinate, and reports
(a) target-side statistics, (b) the implied violation rate under a Gaussian
residual model calibrated to the reported test R^2.
"""
import gzip
import sys

import numpy as np
from scipy.stats import norm

M_PATH = "/data8/junyi/methdata/processed_meth/MC_AMY.CG.m.bedGraph.gz"
H_PATH = "/data8/junyi/methdata/processed_meth/MC_AMY.CG.h.bedGraph.gz"
NMAX = 3_000_000


def read_pairs(nmax):
    m_vals, h_vals = [], []
    with gzip.open(M_PATH, "rt") as fm, gzip.open(H_PATH, "rt") as fh:
        for i, (lm, lh) in enumerate(zip(fm, fh)):
            if i >= nmax:
                break
            am, ah = lm.split(), lh.split()
            if i < 5:  # sanity: coordinates must line up
                assert am[:3] == ah[:3], (am[:3], ah[:3])
            m_vals.append(float(am[3]))
            h_vals.append(float(ah[3]))
    m = np.asarray(m_vals) / 100.0
    h = np.asarray(h_vals) / 100.0
    return m, h


def main():
    m, h = read_pairs(NMAX)
    s = m + h
    print(f"sites                : {len(m):,}")
    print(f"mean  m / h / m+h    : {m.mean():.4f} / {h.mean():.4f} / {s.mean():.4f}")
    print(f"sd    m / h / m+h    : {m.std():.4f} / {h.std():.4f} / {s.std():.4f}")
    print(f"quantiles m+h        : {np.percentile(s, [50, 90, 95, 99, 99.9]).round(3)}")
    for thr in (0.8, 0.9, 1.0):
        print(f"P(m+h > {thr:>4.2f})        : {np.mean(s > thr) * 100:6.3f}%")
    print(f"P(m < 0.02)          : {np.mean(m < 0.02) * 100:6.2f}%  (c/m unstable)")
    hi = m > 0.5
    print(f"among m>0.5 (n={hi.sum():,}): mean m+h = {s[hi].mean():.3f}, "
          f"P(m+h>1) = {np.mean(s[hi] > 1) * 100:.2f}%")

    # Residual model: sigma_res = sigma * sqrt(1 - R^2), R^2 from the 2-head run.
    r2_m, r2_h = 0.7733, 0.6733
    sm = m.std() * np.sqrt(1 - r2_m)
    sh = h.std() * np.sqrt(1 - r2_h)
    print(f"\nsigma_res m / h      : {sm:.4f} / {sh:.4f}")
    print("implied P(m_hat+h_hat>1), independent Gaussian residuals:")
    for rho in (0.0, 0.3, 0.6):
        var = sm ** 2 + sh ** 2 + 2 * rho * sm * sh
        p = norm.sf((1.0 - s) / np.sqrt(var))
        print(f"  rho={rho:.1f}  overall {p.mean()*100:6.3f}%   "
              f"among m>0.5 {p[hi].mean()*100:6.2f}%   "
              f"among m+h>0.9 {p[s>0.9].mean()*100:6.2f}%")
    print("\nNOTE: global sigma assumes homoscedastic residuals; real errors are "
          "larger in high-methylation / repeat regions, so these are lower bounds "
          "for the regions that matter most.")


if __name__ == "__main__":
    sys.exit(main())

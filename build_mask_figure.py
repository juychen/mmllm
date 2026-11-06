import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HOLE = [0, 10, 25, 50, 100]
BP50 = {"5hmc": [0.1701, 0.4791, 0.5741, 0.6259, 0.5860],
        "5mc": [0.3530, 0.5721, 0.6820, 0.7178, 0.6693]}
BP0 = {"5hmc": [0.6601, 0.2084, -0.0827, -0.3680, -0.5061],
       "5mc": [0.7600, 0.2840, 0.0098, -0.1365, -0.1406]}
NAIVE = {"5hmc": [0.245, 0.216, 0.205, 0.177, 0.134],
         "5mc": [0.255, 0.226, 0.219, 0.190, 0.156]}

NAVY, MAROON, GREY = "#002060", "#BD374A", "#8A8F98"
fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.3), sharey=True)
for ax, head, title in zip(axes, ["5hmc", "5mc"],
                           ["5hmC reconstruction", "5mC reconstruction"]):
    ax.axvspan(45, 55, color="#BD374A", alpha=0.10, lw=0)
    ax.plot(HOLE, BP50[head], "-o", color=NAVY, lw=2.4, ms=7,
            label="trained with +/-50 bp mask")
    ax.plot(HOLE, BP0[head], "-s", color=MAROON, lw=2.4, ms=6.5,
            label="trained without neighbourhood mask")
    ax.plot(HOLE, NAIVE[head], "--", color=GREY, lw=1.8,
            label="copy nearest visible CpG")
    ax.axhline(0, color="#444", lw=0.9)
    ax.set_title(title, fontsize=12.5, color=NAVY, loc="left", weight="bold")
    ax.set_xlabel("hidden neighbourhood around each scored CpG (bp)", fontsize=9.5)
    ax.set_xticks(HOLE)
    ax.grid(alpha=0.25, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel("pooled R² on the scored CpGs", fontsize=9.5)
axes[0].legend(fontsize=8.6, frameon=False, loc="lower left")
axes[0].text(51.5, 0.02, "bp50 training protocol", rotation=90, fontsize=7.6,
             color=MAROON, va="bottom", ha="center")
fig.text(0.01, 0.015, "2,938 DMR test windows · 124,136 scored CpGs · "
         "each model evaluated under every hole width (off-protocol points are OOD for that model)",
         fontsize=7.8, color="#555")
fig.tight_layout(rect=[0, 0.04, 1, 1])
fig.savefig("deliverables/mask_regime_compare.png", dpi=200)
print("saved")

import matplotlib.pyplot as plt
import numpy as np

# 数据
target_label = "16384"
pearson_r = 0.790
r2 = 0.624

# 上传图片中的绿色
bar_color = "#52C569"

# 柱子宽度：数值越小柱子越窄
bar_width = 0.22

# x轴位置
x = [0]

fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=300)

yticks = np.arange(0, 1.01, 0.2)

# -------- 第一个图：Pearson r --------
axes[0].bar(
    x,
    [pearson_r],
    width=bar_width,
    color=bar_color,
    edgecolor="black"
)

axes[0].text(
    x[0],
    pearson_r + 0.02,
    f"{pearson_r:.3f}",
    ha="center",
    va="bottom",
    fontsize=10
)

axes[0].set_xticks(x)
axes[0].set_xticklabels([target_label])
axes[0].set_xlim(-0.6, 0.6)
axes[0].set_ylim(0, 1)
axes[0].set_yticks(yticks)
axes[0].set_xlabel("Target length")
axes[0].set_ylabel("Pearson r")
axes[0].set_title("Pearson r")


# -------- 第二个图：R² --------
axes[1].bar(
    x,
    [r2],
    width=bar_width,
    color=bar_color,
    edgecolor="black"
)

axes[1].text(
    x[0],
    r2 + 0.02,
    f"{r2:.3f}",
    ha="center",
    va="bottom",
    fontsize=10
)

axes[1].set_xticks(x)
axes[1].set_xticklabels([target_label])
axes[1].set_xlim(-0.6, 0.6)
axes[1].set_ylim(0, 1)
axes[1].set_yticks(yticks)
axes[1].set_xlabel("Target length")
axes[1].set_ylabel(r"$R^2$")
axes[1].set_title(r"$R^2$")


# 美化
for ax in axes:
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

plt.tight_layout()

plt.savefig("two_barplots_narrow.png", dpi=300, bbox_inches="tight")
plt.savefig("two_barplots_narrow.pdf", bbox_inches="tight")

plt.show()

import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ======================
# 1. 设置总路径和条件顺序
# ======================
base_dir = Path("/data1st2/zhangyr/data/mmllm/modelb/modelb_blocks2")

conditions = [
    "AMY_MC",
    "AMY_MW",
    "HIP_MW",
    "HIP_MC",
    "PFC_MW",
    "PFC_MC",
]


# ======================
# 2. 读取每个 json 文件
# ======================
results = []

for cond in conditions:
    cond_dir = base_dir / cond

    if not cond_dir.exists():
        raise FileNotFoundError(f"Folder not found: {cond_dir}")

    json_files = list(cond_dir.glob("*.json"))

    if len(json_files) == 0:
        raise FileNotFoundError(f"No json file found in: {cond_dir}")

    if len(json_files) > 1:
        raise RuntimeError(f"More than one json file found in: {cond_dir}")

    json_file = json_files[0]

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    result = data["results"][0]
    
    final_val_pearsonr = float(result["final_val_pearsonr"])
    final_val_r2 = float(result["final_val_r2"])

    results.append({
        "condition": cond,
        "final_val_pearsonr": final_val_pearsonr,
        "final_val_r2": final_val_r2,
        "json_file": json_file.name,
    })


df = pd.DataFrame(results)

print(df)


# ======================
# 3. 定义画柱状图函数
# ======================
def plot_bar(df, y_col, y_label, title, save_name):
    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(df["condition"], df[y_col])

    ax.set_xlabel("Condition")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_ylim(0, 1)

    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["condition"], rotation=45, ha="right")

    for bar, value in zip(bars, df[y_col]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    # 保存 png
    png_path = base_dir / save_name
    plt.savefig(png_path, dpi=300, bbox_inches="tight")

    # 保存 pdf
    pdf_path = base_dir / save_name.replace(".png", ".pdf")
    plt.savefig(pdf_path, bbox_inches="tight")

    plt.show()

    print(f"Saved PNG to: {png_path}")
    print(f"Saved PDF to: {pdf_path}")


# ======================
# 4. 画 Pearson r 柱状图
# ======================
plot_bar(
    df=df,
    y_col="final_val_pearsonr",
    y_label="Pearson r",
    title="Final Validation Pearson r",
    save_name="final_val_pearsonr_bar.png"
)


# ======================
# 5. 画 R² 柱状图
# ======================
plot_bar(
    df=df,
    y_col="final_val_r2",
    y_label=r"$R^2$",
    title=r"Final Validation $R^2$",
    save_name="final_val_r2_bar.png"
)

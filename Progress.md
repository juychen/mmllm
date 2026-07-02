# mmllm
mmllm
## EXP-0 the initial toy experiment with Hyena on 5mC and 5hmC tracks, using the same 10k regions as in the original Hyena toy experiment. The goal is to see if Hyena can learn to predict the 5mC track from the 5hmC track, which are known to be correlated but not identical.
### Note:0414 I have tried to tune the initial param, confirming the crosshyena with long_mixer='conv' and filter_len=4. Then add the Hyena layer. The r2 in 3000 data dataset is around 0.4.\
## Note: 0415 I have run the model from 1000 to 70000 data points. The r2 is around 0.5 and pearson r is 0.7 for 70000 data points, which is pretty good for this toy experiment. The model is still underfitting, and I will try to tune the hyperparameters and train for more epochs to see if I can get better performance.
## Note: 0421 I improved the analysis notebook with timestamp-based result lookup, C-only/G-only plots, result-json glob summaries across AMY/HIP/PFC and MC/MW, and grouped Pearson r/R^2 visualization by prediction task and sample size. I also added reproducibility controls by fixing random seeds and exporting seed settings to JSON. I implemented an optional sinusoidal positional encoding switch before CrossHyena, but the current PE design reduced performance at both 5000 and 20000 samples on AMY_MC, so PE is not part of the mainline setting for now. The current next ablation target is reverse-complement augmentation on AMY_MC.

## AMY_MC 实验结果整理 (2026-06-30)

### 实验配置

| 参数 | 值 |
|------|-----|
| 模型 | model_b (CrossHyena fusion, 2 blocks) |
| 输入模态 | 5mC (query) + DNA sequence + ATAC-seq (context) |
| 目标模态 | 5hmC |
| Mask mode | cpg_forward |
| RC augmentation | ✅ 开启 |
| Position encoding | ❌ 关闭 |
| Hidden dim | 64 |
| Target length | 16384 bp |
| Batch size | 2, grad accumulation ×64 (effective 128) |
| Optimizer | AdamW, lr=1e-3, weight_decay=1e-5 |
| Scheduler | Cosine, min_lr=1e-5, patience=15 |
| Max epochs | 100, early stop patience=5 |
| Seed | 7 |

### 实验结果

| 指标 | 2000 DMRs | 80137 DMRs (全量) |
|------|-----------|-------------------|
| **Timestamp** | 2026-06-30-10-59-52 | 2026-06-30-15-38-58 |
| **Train regions** | 3,030 | 127,184 |
| **Val regions** | 970 | 33,090 |
| **Non-overlap groups** | 297 | 10,388 |
| **Best epoch** | 19 | 16 |
| **Best val loss** | 174.40 | 120.63 |
| **Best val R²** | **0.4487** | **0.6719** |
| **Best val Pearson r** | **0.6710** | **0.8198** |

### 关键发现

1. **数据规模 scaling 效果显著**：从 2000 → 80137 DMRs，R² 从 0.45 提升到 0.67 (+49%)，Pearson r 从 0.67 提升到 0.82 (+22%)
2. **模型快速收敛**：全量数据仅需 16 epochs 即达到最佳（early stop），2000 DMRs 需 19 epochs
3. **RC augmentation 有效**：与之前的笔记对比，加上 RC augmentation 后在全量数据上达到了 R²=0.67, r=0.82，是当前最佳结果
4. **单 CpG 分辨率**：模型在 16384 bp 窗口中逐 CpG 预测 5hmC 水平

### 调试历程

AMY_MC 实验经过 6 次失败后成功，问题依次为：
1. `ModuleNotFoundError: pyBigWig` → conda 环境缺少依赖
2. `ModuleNotFoundError: pyfaidx` → 同上
3. `ModuleNotFoundError: pysam` → 同上
4. ATAC bigWig 路径错误（`/data2st2/...` 应为 `/data1st1/...`）
5. 染色体名重复前缀 `chrchr1`（bed 文件已有 chr，pyfaidx 又加了一次）

### 当前并行实验状态 (2026-07-02)

| 数据集 | 状态 | 备注 |
|--------|------|------|
| **AMY_MC** | ✅ 完成 | 2000 + 80137 DMRs |
| **AMY_MW** | ⚠️ 部分完成 | 01:44 启动，有 checkpoint (best_80137.pt, 25.4M) 但 log 为空，需确认 |
| **HIP_MC** | 🔄 运行中 | 11:09 启动 |
| **HIP_MW** | 🔄 运行中 | 11:09 启动 |
| **PFC_MC** | 🔄 运行中 | 11:09 启动 |
| **PFC_MW** | 🔄 运行中 | 11:09 启动 |

所有并行实验使用 `run_all_m5c_modelb.sh cross_hyena` 启动，统一配置：model_b + CrossHyena, RC aug, cpg_forward mask, 100000 DMRs.

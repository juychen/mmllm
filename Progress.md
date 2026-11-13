## 2026-11-13 · 洞宽协议收尾 + 能力曲线

### A/B 两条 100 ep 从零训练（同 split、同 seed、同 loss 口径，均遮 ~15% CpG）
| 协议 | test 5mC R2/rho | test 5hmC R2/rho |
|---|---|---|
| A 预算式随机洞宽 | 0.7627 / 0.8741 | 0.6661 / 0.8156 |
| B 预算式固定 50 bp | 0.7549 / 0.8692 | 0.6662 / 0.8157 |

val 上二者相差 0.0005（噪声级）；但换成**评测洞宽**后分出高下（见下）。

### 能力曲线（2,938 DMR 窗，pooled R2，5hmC / 5mC）
| 评测洞 | A 随机 | B 固定50 |
|---|---|---|
| single | **0.6406 / 0.7567** | 0.5847 / 0.6894 |
| 0 bp | **0.6503 / 0.7575** | 0.5908 / 0.6844 |
| 50 bp | 0.6134 / 0.7177 | **0.6170 / 0.7102** |
| 100 bp | 0.5684 / 0.6649 | 0.5638 / 0.6372 |

结论：**val（固定 ±50）掩盖了泛化差异**。A 在窄洞上高 0.056（5hmC）/0.067（5mC），曲线更平；B 只在自己训练过的 50 bp 上略占优（0.0036，噪声级）。此前"随机洞宽无收益"的结论需修正为：**分布内无收益，分布外有收益**。

### 区域 run（bp0 主协议，100 ep）
AMY_MW 0.7098 / HIP_MW 0.6778 / HIP_MC 0.6666（5hmC val）；5mC 均 0.80-0.81。组织效应（0.03-0.04）远大于协议效应（0.008）。

### 仍在跑
六组池化 bp0（ep16/30，约 4.5 h/ep，ETA 11-16）；PFC_MW / PFC_MC 未启动（队列守护进程按顺序等待，GPU 1/2/5 空转）。

# mmllm

## 2026-11-03 · 最新进展（置顶）

> **口径先看这里**：11-03 起同时换了**掩码规则**和**test 集定义**，新数字不能直接和 10-23/10-26 的 0.797 / 0.688 比。
>
> - **掩码**：从「随机遮 15% 的单 CpG」改为「随机遮 15% 的 **CpG 对**，隐藏 C/G ±50 bp 邻域，loss 只算 C」（commit `00e9e54` "ADD strict masking"，见 [data.py](data.py) `generate_cpg_pair_mask`）。相邻 CpG 的信息被遮掉，**任务变难**。
> - **test 集**：改为显式 `output/holdout_testchroms/cpg_test.bed`（7,412 窗口：chr8 尾 640 / chr9 4,422 / chrX 2,290 / chrY 60）。已核对：它与旧 `auto_3way` 产出的 `test_regions.bed` **逐窗口完全一致**，所以历史 test 数字仍然可比。
> - **val 集**：带 `--test-bed` 时 val = 非 test 池的 20%（15,845 窗口），旧 `auto_3way` 是 9,520 → **val 数字不可直接比**。

### 一、相对上一版（2026-10-23 汇报）改了什么

| 项 | 上一版（10-23 ~ 10-26） | 本版（11-03） |
|---|---|---|
| 掩码 | 单 CpG，15% | **CpG 对 + ±50 bp 邻域隐藏，loss 只在 C** |
| test 集 | `auto_3way` 尾部（隐式，事后才知道是 chr8尾/9/X/Y） | **显式 holdout BED**，窗口集合与旧 test 完全相同 |
| val 集 | 9,520（10%） | 15,845（带 test-bed 时 = 非 test 池的 20%） |
| 2-head 主线 | 30 ep（10-07：val 0.688） | **100 ep × 3 条**（drop 0.0 / 0.1 / 0.2），进行中 |
| 数据规模 | cCRE 80,137 | 另加 **全基因组 pool 142,217**、**6 组池化 379,230** |
| 调参 | 无 | **`parametest.py`**：独立 5-epoch 筛查框架 |

### 二、已完成的结果（唯一跑完的长跑）

**4-head 100ep**（旧单-CpG 掩码、`auto_3way`、no-drop）：2026-10-26 10:02 → 11-02 00:54，**best epoch 98**

| 指标 | 5mc | 5hmc | ATAC | RNA |
|---|---|---|---|---|
| val R² | 0.797 | 0.688 | 0.980 | 0.937 |
| test R² | 0.773 | 0.667 | 0.978 | 0.935 |
| test Pearson | 0.881 | 0.818 | 0.991 | 0.968 |

- 同一个 run 从 30 ep 到 98 ep：val **+0.012**（5mc）/ **+0.016**（5hmc），代价是 2 天 → 6.6 天 → **收益明显递减**
- 对比 2-head all-others（10-23 汇报）：test 5mc 0.773 vs 0.773（持平），5hmc 0.667 vs **0.673**（四头反而略低）→ 四个头没有给甲基化带来净收益

### 三、进行中（11-03 起跑，均在新掩码下）

| 实验 | 进度 | 当前 val R²（5mc / 5hmc） | 每 epoch | 预计完成 |
|---|---|---|---|---|
| AMY_MC 2-head drop0.1 | ep5/100 | 0.699 / 0.585（ep4） | 40.6 min | 11-06 上午 |
| AMY_MC 2-head drop0.0 | ep3/100 | 0.641 / 0.507（ep2） | 40.7 min | 11-06 中午 |
| AMY_MC 2-head drop0.2 | ep3/100 | 0.605 / 0.502（ep2） | 40.7 min | 11-06 中午 |
| 全基因组 train pool（142,217 DMRs） | ep2/100 | 0.030 / 0.139（ep1） | 76 min | 11-08 夜间 |
| 6 组池化 2-head 30ep | ep1 | — | ~5.1 h（估算） | ~11-10 |

- 新掩码下 ep1 就拿到 0.60 / 0.49，是**任务变难后的正常水平，不是退化**；要和旧的 0.797/0.688 对比，必须等 ≥30 ep 的新基线
- 全基因组 run 的 ep1（0.03 / 0.14）低得多，因为窗口不再富集在 cCRE 上，任务本身更难

### 四、调参框架与一条协议结论

新增 [parametest.py](parametest.py)：独立脚本，只 import `data.py`/`models.py`/`utils.py`，自己起训练循环；5 epoch；**train/val/test 三段都构建、都评测、都落盘 BED**；val mask 按窗口身份定死（`crc32(chr:start)^seed`），因此与 batch size / 梯度累积无关，配置之间可比。

实测结论（**建议写进方法学**）：**短筛查的预算要按 optimizer step 算，不能按 epoch 算。**

- 生产脚本 2-head 1 epoch = 56,880×2(RC) ÷ 32 ÷ 8 = **444 个 optimizer step**
- 代理集 12k × 5 epoch = 只有 **340 步**，模型还停在「预测均值」，val R² = **0.002**
- 而且 5-epoch 的 cosine 会在步数用完之前就把 lr 衰减到 `min_lr`，等于把后半程的学习率浪费掉

对策已写进脚本默认值：screening 强制 constant lr（`--force-scheduler none`），代理集提到 20k（≈560 步）。当前 sweep 已按要求停止，没有配置跑完；脚本与 split BED 保留，一条命令可恢复。

### 五、下一步

1. 等 drop 0.1 / 0.0 / 0.2 三条跑到 ≥30 ep，建立**新掩码下的 2-head 基线**，再谈是否相对 0.797/0.688 有提升
2. 若新掩码确实更难，考虑 mask_fraction 0.15 → 0.10，或把 ±50 bp 邻域收窄，作为独立消融
3. 全基因组 pool 的 ep1 明显落后，建议先用 30k 子集确认学习率是否合适，再烧 100 ep
4. 6 组池化 5.1 h/epoch × 30 ep ≈ 6.4 天，是否值得跑满 30 ep 需要单独决策（可先用 2 组做 pilot）

---

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

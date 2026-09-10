# Figure Plan v2 — CrossHyena Multimodal 5hmC Prediction

**Locked down: 2026-09-10**

本规划假设：

- Fig 1（不计入 4 张大图）= 概念图 + **完整版** CrossHyena 架构 + 输入/输出/数据来源标注
- 因此 Fig 2 **不再画任何架构细节**，避免与 Fig 1 重复
- 公共数据集（GSE214845、ENCODE ATAC、oxBS-derived 5hmC 等）已纳入主线叙事

---

## 总体叙事弧线

| 图 | 章节归属 | 角色 | 回答问题 | 是否含架构 |
|---|---|---|---|---|
| 1（不算） | Intro 末尾 | 概念 + 完整架构 + 数据 | 任务是什么、为何需要多模态 | ✅ |
| **2** | Method §1 | 数据 pipeline + 训练策略 | 数据怎么来、怎么训、怎么避免泄露 | ❌（仅 input encoding 细节） |
| **3** | Results §1 | Main results | 精度如何、可迁移性如何 | — |
| **4** | Results §2 | Ablation（AMY_MC 主） | 每个组件贡献多大 | — |
| **5** | Results §3 | Interpretation | 学到了什么生物学意义 | — |

---

## Figure 1（概念图 + 完整架构）

> **不计入 4 张大图**，但需作为锚点。本规划中 Fig 2–5 的范围**严格不与 Fig 1 重复**。

**Fig 1 必须包含**：
- 任务图示：5mC + DNA + ATAC → CrossHyena → 5hmC
- CrossHyena **完整架构**（含 block 内部 long-conv + gated MLP + cross-attn）
- 输入/输出维度标注 `(B, L, C)`
- 数据来源标注（internal + public）

**Fig 1 不包含**（留给 Fig 2）：
- 数据区域采样与扩张
- non-overlap group 划分细节
- mask 策略对比
- RC augmentation
- optimizer / scheduler 配置
- loss 公式

---

## Figure 2 — Data pipeline + Training strategy（7 panel）

> 整张图从左到右贯穿：**数据 → 区域/编码 → split → mask → 增强 → 训练配置 → loss**。

| Panel | 标题 | 内容 | 关键要素 |
|---|---|---|---|
| **a** | Region sampling & DMR expansion | DMR metadata → 中心对齐 → 16384 bp 固定窗口；长度不足的扩展策略 | 标注窗口大小、中心化方法；与公共数据同长度采样保持一致 |
| **b** | Non-overlap group split | 用 group ID 切 train/val (80/20)；公共数据按样本/鼠 ID 独立 group | 用一张 schematic 图说明"同组 DMR 不跨集合"；列出 6 cell 的 group 数 |
| **c** | Input encoding（非架构细节） | ① DNA one-hot (A/C/G/T) → linear embed ② 5mC per-CpG scalar → linear ③ ATAC bigWig → per-position scalar；C/T strand 区分；PE 开关（说明当前关闭） | 公式 + 小示意；**不含** CrossHyena block 内部 |
| **d** | CpG mask strategies | `cpg_forward` / `cpg_both` / `all` 三种 mask 在一个示例 region 上的可视化对比 | 在同一坐标轴上画三条 mask 标记；标注 main 选 `cpg_forward` 的理由（与 5hmC 真实读出位置匹配） |
| **e** | RC augmentation & effect | 训练时随机 reverse-complement；训练曲线 on/off 对比 | 与你笔记里的 R² 0.45 → 0.67 提升对应；**这是 Fig 4-f 的预告** |
| **f** | Training configuration | AdamW (lr=1e-3, wd=1e-5) + cosine schedule + early stop (patience=5, max 100 epoch) | 表格 + 一段说明文字 |
| **g** | Loss function | masked MSE：只对 `cpg_forward` 位置反传 | 公式 + 与 vanilla MSE 的对比示意 |

**Fig 2 范围红线**：
- ❌ 不画 CrossHyena block 内部（Fig 1 已覆盖）
- ❌ 不画数据来源总览（Fig 1 已覆盖）
- ❌ 不画训练曲线（留给 Fig 3-b）

---

## Figure 3 — Main results（7 panel）

| Panel | 标题 | 内容 |
|---|---|---|
| **a** | Headline metrics heatmap | 6 cell (AMY/HIP/PFC × MC/MW) 的 R² + Pearson r 双指标；标注全量 vs 2k DMR 对比 |
| **b** | Training curves | loss / R² vs epoch，6 条线按脑区配色，标出 early-stop 点 |
| **c** | Data scaling | R² / Pearson r vs #DMR (1k→70k) 主图 AMY_MC；其他 5 cell 用 facet 小图 |
| **d** | Pred vs Obs scatter | 6 cell 的散点图（2×3 faceted），含 y=x 参考线、整体指标标注 |
| **e** | Genome browser examples | 3 个典型 DMR，每个 4 行：5mC input / 5hmC 真值 / 5hmC 预测 / ATAC；**每个 DLR 双行对比：内部 vs 公共同区域真值** |
| **f** | Cross-dataset generalization | 在公共 BS-seq（GSE214845）、oxBS-derived 5hmC、ENCODE ATAC 上的 R² + Pearson r；标注数据来源与覆盖度 |
| **g** | Cross-region × cross-condition × cross-dataset matrix | 6 cell × {internal, GSE214845, oxBS-Lister} = 6×3 迁移矩阵；展示模型在异源数据上的稳健性 |

---

## Figure 4 — Ablations on AMY_MC（7–8 panel）

| Panel | 标题 | 内容 |
|---|---|---|
| **a** | Modality ablation | bar chart：full / drop 5mC / drop ATAC / drop seq / seq-only / 5mC-only；ΔR² + ΔPearson r |
| **b** | Sequence-only vs multimodal | 散点 + 柱状图，证明多模态相对纯序列的边际收益 |
| **c** | ATAC 边际贡献 | `5mC+seq` vs `5mC+seq+ATAC` 学习曲线叠加 |
| **d** | Window length ablation | R² vs target_length (1024 / 4096 / 8192 / 16384) + compute cost |
| **e** | Mask mode comparison | `cpg_forward` vs `cpg_both` vs `all` 的 R²/Pearson 对比柱状图 |
| **f** | RC augmentation effect | on/off 学习曲线对比（接 Fig 2-e 的预告，给出完整版） |
| **g** | 架构对比 | CrossHyena-fusion vs 单分支 Hyena vs MLP（接 Fig 1 的架构创新点，给性能验证） |
| **(h)** | **训练数据来源 ablation** | internal-only / public-only / internal+public / internal→public fine-tune 的 R² 曲线 |

> Panel g 与 h 可合并成 2×2 拼图以节省空间。

---

## Figure 5 — Interpretability + 生物学解读（7 panel）

| Panel | 标题 | 内容 |
|---|---|---|
| **a** | Cross-attention heatmaps | 2–3 个 example DLR，画 query CpG 对 context 位置的 attention 矩阵 + gene annotation overlay |
| **b** | Attention span distribution | 全验证集统计：attention weight 随 genomic distance 的衰减曲线（合并 internal + public） |
| **c** | Saliency / in-silico mutagenesis | 对 5mC 输入逐位扰动看 Δprediction；定位关键 CpG |
| **d** | Motif enrichment | 高 attention CpG ±100 bp 的 de novo motif；与 JASPAR 2024 比对；与公共 cCREs 重叠统计 |
| **e** | Genomic annotation stratification | 按 ChromHMM 15-state 分层统计 5hmC 预测误差；用 ENCODE mouse annotations |
| **f** | Case study | 1 个已知增强子/启动子 DLR，详细展示 attention + saliency + 预测；用箭头注释关键 CpG |
| **g** | 5mC vs 5hmC error correlation | scatter：per-region 5mC prediction error vs 5hmC prediction error；回到 paper 的科学动机（5hmC ≠ 5mC，需要多模态） |

---

## 公共数据集接入清单

### 必须下载/处理的

| 数据集 | 类型 | 用途 | 优先级 |
|---|---|---|---|
| **GSE214845** | mouse brain WGBS（5mC） | Fig 3-f 跨数据集验证 | P0 |
| **ENCODE mouse brain ATAC** | ATAC bigWig | Fig 3-f ATAC 跨来源测试 | P0 |
| **GSE112520** | mouse ESC oxBS（5mC / 5hmC 分开） | Fig 3-f + Fig 5 平台对照 | P1 |
| **Lister 2013** | human ESC oxBS | Fig 4-h cross-species fine-tune（可选） | P2 |
| **ENCODE mouse brain cCREs** | 注释 | Fig 5-d/e | P1 |
| **JASPAR 2024** | motif 数据库 | Fig 5-d 比对 | P0 |
| **Roadmap ChromHMM** | 15-state 注释 | Fig 5-e | P0 |

### 接入流程（必须先做的预处理）

1. **基因组版本统一**：所有公共数据 liftOver 到 mm10（你的脚本默认 GRCm38.p6）
2. **Coverage correction**：5hmC = BS − oxBS 会出现负值/零值，需用 coverage 加权修正
3. **样本独立性切分**：按 sample/mouse ID 重新 group，避免同只鼠的 DMR 跨 train/val
4. **统一 CpG 注释**：用 UCSC mm10 CpG islands 重新 mask
5. **格式统一**：所有 bedGraph → bigWig；chromosome naming 一致

---

## 实施 task 跟踪

下面这些 task 在你准备开干时可以创建：
1. 公共数据集下载 + liftOver（`download_public.sh` + `liftover_public.sh`）
2. 5hmC BS−oxBS 推导 + coverage correction（`derive_5hmc.py`）
3. 公共数据纳入 non-overlap group 划分（`merge_groups.py`）
4. Fig 2 panel 脚本（每个 panel 一个 `fig2_*.py` 或一个 notebook）
5. Fig 3 panel 脚本（`fig3_*.py`）
6. Fig 4 panel 脚本（`fig4_*.py`）
7. Fig 5 panel 脚本（`fig5_*.py`）
8. 全部 figure 合成（`assemble_figure*.py`，合成 PDF/EPS）

---

## Checklist（写作前最后一遍过）

- [ ] Fig 1 已画好完整架构（确认 cross-attn、long-conv、gated MLP 顺序）
- [ ] Fig 1 包含数据来源标注
- [ ] Fig 2 不画任何架构细节
- [ ] Fig 3-e 每个 DLR 双行：内部 + 公共
- [ ] Fig 3-f / Fig 4-h 公共数据集已处理完毕
- [ ] Fig 4 panel 数 ≤ 8（合并 g/h）
- [ ] Fig 5-d/e 用了 ENCODE 注释
- [ ] 全部 figure 用统一色板（colorblind-safe，参考 ColorBrewer Set2）
- [ ] 全部 figure 导出 PDF + EPS 矢量（投稿用）
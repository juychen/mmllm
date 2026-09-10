# 公共数据集可用性评估报告

**生成时间**: 2026-09-10
**数据根目录**: `/data1st1/junyi/methdata/`
**评估方式**: 实际遍历 + pyBigWig 验证染色体命名与值域

---

## 0. mmllm 内部数据（已用，排除在外）

```
/data2st1/junyi/output/llm0401/processed_meth/   → 5mC/5hmC bedGraph.gz (AMY/HIP/PFC × MC/MW)
/data2st2/junyi/output/atac1112/tobiasbam/BULK/corrected/  → ATAC bigWig
```

> 6 cell (AMY/HIP/PFC × MC/MW)，ChromHMM-like 命名 (`MC_AMY.CG.m.bedGraph.gz` 等)，mm10。

---

## 1. 已下载数据集清单（按可用性排序）

### ✅ TIER 1 — 完整可用，建议立即接入

| GSE | 内容 | 5mC | 5hmC | ATAC | 验证结果 |
|---|---|---|---|---|---|
| **GSE214845** | mESC germ layer bifurcation (oxWGBS+ATAC+scRNA+ChIP) | ✓ BS.bw (57 chroms) | ✓ 5hmC.bw (57 chroms, 值域 -0.8~+0.8, BS−oxBS 推导) | ✓ ATAC.bigwig | **格式完美，可直接用**；需过滤 GL456* scaffolds |
| **GSE166423** | Purkinje neurons (WGBS+TAB-seq+ATAC+ChIP) | ✓ purkinje_adult_methyl_CpG_merged.bw | ✓ purkinje_adult_hydroxy_CpG_merged.bw (TAB-seq 直接测) | ✓ broadPeak 多个 | **金标准**：TAB-seq 直接给 5hmC（非推导），无负值 |
| **GSE140125** | DeepH&M cerebellum/cortex | ✓ bismark.cov.gz (chr1... 标准 mm10) | ✓ bismark.cov.gz (TAB-seq, 同一格式) | ❌ 无 ATAC | **全 mm10 坐标**，每文件已 paired WGBS+TAB；适合 cerebellum validation |
| **GSE174048** | 骨骼发育 (WGBS + 5hmC + ATAC) | ✓ methratio.bw (chr 数字命名, **需加 chr 前缀**) | ⚠️ 不在 root，需查 GSE174047 子目录或重新下载 | ✓ narrowPeak.gz (GSE174045) | **坐标系问题**：chrom 用 `1,10,11...,MT,X,Y`，**必须加 chr 前缀**才能和 mmllm 对齐 |
| **GSE231928** | TOP-Seq 同时测 5mC+5hmC+ATAC (单 assay) | ✓ GC_table.tsv.gz | ✓ hmC_table.tsv.gz | ✓ regions_idT.tsv.gz | **唯一同时同 assay 数据**；TSV 格式需解析为 bedGraph |
| **GSE150964** | iPSC reprogramming (WGBS + oxWGBS + ATAC) | ❌ 不在下载中 | ✓ narrowPeak.gz 28 个 (peak) | ✓ narrowPeak.gz 28 个 | 仅 ATAC peaks 已下载；5mC/5hmC 待补 |
| **GSE174048 + GSE189655** | 同上 ATAC bigWig 子目录 | ✓ (见上) | — | ✓ TCKO/WT merge bws | 同上 |

### ✅ TIER 2 — 部分可用，需补充下载

| GSE | 内容 | 状态 |
|---|---|---|
| **GSE244251** | Nanopore 神经细胞 (cortical neurons/astrocytes/microglia) | ✓ `*.bed.gz` 已下载，每文件含 5mC + 5hmC 同一位置 + p-value + FDR；**单 base Nanopore 黄金标准** |
| **GSE279860** | Nanopore duplex cerebellum | ✓ masked.bed.gz 已下载；类 Nanopore 5mC+5hmC 注释 |
| **GSE214830** | oxWGBS mESC (TET1 d2) | ✓ `WT.5hmC.bw` + `WT.5mC.bw` 直接给；无 ATAC |
| **GSE248149** | oxRRBS cancer cell line | ✓ bedGraph.gz 已下载；**track-ready**，chr start end value 格式直接 tabix 用 |
| **GSE212634** | OSCC WGBS+oxWGBS | ✓ MergedCG txt.gz 已下载；含 mC + hmC 两套 |
| **GSE296587** | 6-base CUT&Tag mESC (H3K27ac) | ✓ hmc.bw + mc.bw 直接给 |
| **GSE288331** | hMeDIP Nanopore cerebellum | ✓ narrowPeak.gz + peak_pileup.bed.gz 已下载 |
| **ENCODE_forebrain_ATAC** | ENCODE mouse forebrain ATAC | ✓ 3 个 bigWig (ENCFF424SNT/588MMH/995ODT)，mm10 |
| **ENCODE_mouse_cerebellum_ATAC** | ENCODE mouse cerebellum ATAC | ✓ 2 个 bigWig (ENCFF324THC/674MTJ) |

### ⚠️ TIER 3 — 不推荐/待评估

| GSE | 内容 | 状态 |
|---|---|---|
| GSE103470 | ATAC (mouse ESC 多时点) | ✓ 9 个 .bw，但只有 ATAC，**无 5mC/5hmC** |
| GSE186357 | 胚胎 WGBS+oxWGBS (N14/PBAT) | ✓ CpG.bw 已下载；早期胚胎样本，与你 brain AMY/HIP/PFC 不匹配 |
| GSE247534 | OGT mESC KO (WGBS+oxWGBS) | 仅 RAW.tar，需解压 |
| GSE301936 | HBQ 膀胱 | ✓ txt.gz，但样本类型与你 brain 区域不符 |
| GSE290585 | EM-seq BeadArray 534 samples | ✓ matrix + idat 已下载；**array 数据**，与你的 mm10 DMR 不直接兼容（需 probe→coord mapping） |
| GSE267937 | PD 脑 oxBS EPIC | 同上，**array 数据**，与 mm10 DMR 不兼容 |
| GSE97988 | hES-pancreatic BS+CMS+ATAC | 仅 RAW.tar (26G)，需解压；**人细胞，与你 mouse 数据不直接可用** |
| GSE141152 | Treg multi-omics | RNA counts + ATAC peaks + 5mC.bw (GSE141151) 已下载；5mC 在子目录，ATAC peak 已有 |

---

## 2. 关键 sanity check 结果

### 染色体命名（必须预处理！）

| 数据集 | 命名 | 与 mm10 (chr1...) 对齐方式 |
|---|---|---|
| mmllm 内部 / GSE166423 / GSE140125 / GSE244251 / ENCODE ATAC / GSE296587 / GSE212634 / GSE288331 | ✅ `chr1, chr10...chrM, chrX, chrY` | 直接用 |
| GSE214845 (5hmC/BS/oxBS bws) | 22 chr-prefix + 35 GL456* scaffolds | 用 mm10 chrom sizes 过滤 GL456* |
| **GSE174048 methratio.bw** | ❌ **数字命名** `1, 2, ..., MT, X, Y` | **必须 sed 加 `chr` 前缀** |
| GSE214830 (5mC/5hmC bws) | 22 chr-prefix + 34 scaffolds (chrUn_*, GL456*) | 过滤 scaffolds |
| GSE186357 (CpG bw) | 21 chr-prefix (仅主染色体) | 直接用 |

### 值域（决定 5hmC 是否需要 coverage correction）

| 数据集 | 5hmC 范围 | 5mC 范围 | 解读 |
|---|---|---|---|
| GSE214845 5hmC bw | **-0.80 ~ +0.80** (有负值!) | 0.000 ~ 1.000 | **BS − oxBS 推导**，含负值；**必须 coverage correction** |
| GSE166423 hydroxymethylCpG bw | 0.000 ~ 1.000 (TAB-seq 直接) | 0.000 ~ 1.000 | **TAB-seq 直接测**，**金标准**，无负值 |
| GSE214830 WT.5hmC bw | **-0.80 ~ +0.71** (有负值) | 0.000 ~ 1.000 | **oxWGBS 推导**，含负值，需 correction |
| GSE248149 bedGraph.gz | 0.000 ~ 1.000 (oxRRBS 推导) | 0.000 ~ 1.000 | 单 base bedGraph，trace 校正 |
| GSE296587 H3K27ac hmc bw | 0.000 ~ 30.000 (count-like) | 0.000 ~ 55.000 | 6-base CUT&Tag，**count 不是 methylation ratio**；需 /coverage |
| GSE231928 hmC table | 待解析 | 待解析 | TSV，需先看 schema |

### 关键警告（你 memory 笔记里已记过）

> **BS − oxBS = 5hmC**（不是 5mC！）
> BS 测的是 5mC + 5hmC，oxBS 测的是 5mC only。
> 推算出的 5hmC 含大量**负值/零值**，必须 coverage-based correction。

已经在 `docs/5hmc_from_bs_oxbs.md` 中记录。

---

## 3. 对 Fig 3 / Fig 4 的具体建议

### Fig 3-f 跨数据集泛化（推荐组合）

| 角色 | 数据集 | 用法 |
|---|---|---|
| **主 external mouse** | **GSE166423** | TAB-seq 5hmC 直接测；用 Purkinje neuron 5hmC bw 验证模型 |
| **同物种同 assay 交叉验证** | **GSE214845** | 5 个样本 (CM59, CM69, KO15, WT70, WT72) 都有 5hmC + BS + oxBS + ATAC；可做 leave-one-sample-out |
| **同 assay 同物种** | **GSE140125** | bismark.cov 直接读，cerebellum 验证 |
| **Nanopore 黄金对照** | **GSE244251** | 高质量 5hmC，可做"模型 vs Nanopore"基准 |

### Fig 4-h 训练数据来源 ablation

| 实验组 | 数据组合 |
|---|---|
| internal only | 当前 6 cell |
| public only | GSE166423 + GSE140125 + GSE214845 |
| internal + public 联合 | 混合训练，公共样本作为增强 |
| internal → public fine-tune | 先在 internal pretrain，再在 public fine-tune |

### Fig 5-d/e 注释增强

| 用途 | 数据集 |
|---|---|
| Genomic annotation (Fig 5-e) | ENCODE mouse cCREs (需补下) + ChromHMM |
| Motif enrichment (Fig 5-d) | JASPAR 2024 |

---

## 4. 必须先做的预处理脚本

### 4.1 染色体命名归一化（`scripts/public_data/normalize_chroms.py`）

```python
# 输入：GSE174048 的 methratio.bw（数字命名）
# 输出：chr1, chr2... chrM, chrX, chrY
# 方法：用 pyBigWig → pyBigWig 或 bigWigToBedGraph + awk sed
```

### 4.2 Scaffolds 过滤（`scripts/public_data/filter_scaffolds.py`）

```python
# 输入：GSE214845 / GSE214830 的 bw（含 GL456*, chrUn_*）
# 输出：仅 chr1-chrM, chrX, chrY 的新 bw
# 方法：使用 mm10.chrom.sizes 白名单过滤
```

### 4.3 BS−oxBS 5hmC 负值修正（`scripts/public_data/correct_5hmc_negative.py`）

```python
# 输入：GSE214845 5hmC bw（负值）
# 输出：coverage-corrected 5hmC bw
# 方法：
#   1) 计算每个 CpG 的 BS coverage + oxBS coverage
#   2) 5hmC = BS - oxBS
#   3) coverage < threshold 时设 5hmC = 0（噪声）
#   4) coverage >= threshold 时保留原值
#   5) Clamp 到 [0, 1]
# 参考你 memory 里的 5hmc-from-bs-oxbs.md
```

### 4.4 BedGraph ↔ BigWig 转换（`scripts/public_data/convert_bw.sh`）

```bash
# bismark cov / MergedCG txt.gz → bedGraph → bigWig
zcat GSE140125/*.bismark.cov.gz | awk '{OFS="\t"}{print $1,$2,$3,$4}' | sort -k1,1 -k2,2n | bgzip > tmp.bedGraph.gz
tabix -p bed tmp.bedGraph.gz
bedGraphToBigWig tmp.bedGraph mm10.chrom.sizes tmp.bw
```

### 4.5 Non-overlap group 重划分（`scripts/public_data/regroup_public.py`）

```python
# 输入：公共数据集的所有 5hmC/5mC bw
# 输出：新的 group ID（按 sample/mouse 划分），避免 train/val 泄露
```

---

## 5. 数据集可用性速查（mmllm 用）

### 5hmC 可用 bw/bedGraph（已下载）
| 文件 | 路径 | 评估 |
|---|---|---|
| purkinje_adult_hydroxy_CpG_merged.bw | `GSE166423/` | ⭐⭐⭐⭐⭐ TAB-seq direct |
| purkinje_p0_hydroxy_CpG_merged.bw | `GSE166423/` | ⭐⭐⭐⭐⭐ |
| purkinje_p7_hydroxy_CpG_merged.bw | `GSE166423/` | ⭐⭐⭐⭐⭐ |
| GSE214845_*_5hmC.bw (5 files) | `GSE214845/` | ⭐⭐⭐ BS−oxBS 推导，需 correction |
| GSE214830_*.5hmC.bw | `GSE214830/` | ⭐⭐⭐ oxWGBS 推导，需 correction |
| GSE296587_H3K27ac.hmc.10x.bw | `GSE296587/` | ⭐⭐ 6-base CUT&Tag (count, 非 ratio) |
| GSM7906421_MDA_MB_231_5mC5hmC.bedGraph.gz | `GSE248149/` | ⭐⭐⭐⭐ oxRRBS direct |
| GSM6542052_OC1_MergedCG_hmC_10x.txt.gz | `GSE212634/` | ⭐⭐⭐ oxWGBS direct |
| GSM4154656_7w_cerebellum_rep1_tabSeq.bismark.cov.gz | `GSE140125/` | ⭐⭐⭐⭐ TAB-seq direct |

### 5mC 可用 bw/bedGraph（已下载）
| 文件 | 路径 | 评估 |
|---|---|---|
| purkinje_adult_methyl_CpG_merged.bw | `GSE166423/` | ⭐⭐⭐⭐⭐ direct |
| GSE214845_*_BS.bw + oxBS.bw | `GSE214845/` | ⭐⭐⭐⭐⭐ direct |
| GSM5285240_WT_methratio.bw | `GSE174048/` | ⭐⭐⭐⭐ direct (但 chrom 需加 chr) |
| GSM4154655_7w_cerebellum_rep1_wgbs.bismark.cov.gz | `GSE140125/` | ⭐⭐⭐⭐⭐ direct |
| GSM7906420_MDA_MB_231_5mC.bedGraph.gz | `GSE248149/` | ⭐⭐⭐⭐⭐ direct |
| GSM6542050_OC1_MergedCG_mC_10x.txt.gz | `GSE212634/` | ⭐⭐⭐⭐⭐ direct |

### ATAC 可用 bw/bedGraph（已下载）
| 文件 | 路径 | 评估 |
|---|---|---|
| ENCFF424SNT/588MMH/995ODT.bigWig | `ENCODE_forebrain_ATAC/` | ⭐⭐⭐⭐⭐ ENCODE 黄金 |
| ENCFF324THC/674MTJ.bigWig | `ENCODE_mouse_cerebellum_ATAC/` | ⭐⭐⭐⭐⭐ |
| GSE214845_*_ATAC.bigwig | `GSE214845/` | ⭐⭐⭐⭐⭐ 同一样本 |
| ATAC_*.broadPeak | `GSE166423/` | ⭐⭐⭐⭐ |
| GSM4563059_T2DMEF1_peaks.narrowPeak.gz | `GSE150964/` | ⭐⭐⭐⭐ |
| GSM2535464_ATAC_B.rmdup.bw | `GSE103470/` | ⭐⭐⭐ ESC 多时点 |
| GSM5662815_N14_PBAT_MII_WT_sc1.CpG.bw | `GSE186357/` | ⭐⭐⭐ 胚胎数据 |

---

## 6. 优先级建议（给 Fig 3-f / Fig 4-h）

### 必须做的（Fig 3-f 撑得起来）

1. ✅ GSE166423 — Purkinje neurons 5hmC bw (TAB-seq direct)
2. ✅ GSE140125 — cerebellum bismark cov (TAB-seq direct)
3. ✅ GSE214845 — mESC germ layer 5hmC bw (BS−oxBS, 需 correction)

### 加分项（Fig 4-h 撑得起来）

4. GSE244251 — Nanopore 神经细胞 bed.gz (单 base 黄金)
5. ENCODE forebrain/cerebellum ATAC — 跨来源 ATAC 测试

### 不必现在做（等 Fig 5 阶段）

6. GSE174048 — 需先做 chrom 前缀修正
7. GSE150964 — 需重新下载 5mC/5hmC tracks
8. GSE290585 / GSE267937 — array 数据需 probe mapping

---

## 7. 推荐立即执行的命令

```bash
# 1) 写 chromosome 归一化脚本
cat > scripts/public_data/normalize_chroms.py << 'EOF'
[脚本内容]
EOF

# 2) 写 BS−oxBS 5hmC 修正脚本
cat > scripts/public_data/correct_5hmc_negative.py << 'EOF'
[脚本内容]
EOF

# 3) 写 scaffolds 过滤脚本
cat > scripts/public_data/filter_scaffolds.py << 'EOF'
[脚本内容]
EOF
```

需要我直接开始写这些预处理脚本骨架吗？还是先帮你从 Tier 1 三个数据集里挑 1 个跑通完整 pipeline 作为模板？
# 公共数据集可用性统计报告（基于 CSV + 实测文件）

**生成时间**: 2026-09-10
**数据源**: `/data1st1/junyi/methdata/GSE_5hmC_5mC_datasets.csv`（75 行）+ 实际文件验证

---

## 1. 总览数字

| 维度 | 数量 |
|---|---|
| CSV 总数据集数 | **75** |
| 已下载 | **20** (26.7%) |
| 未下载 | 55 (73.3%) |
| Mouse | 44 |
| Human | 28 |
| Mouse+Human | 3 (GSE197740 等) |

## 2. Mouse 已下载细分（**核心数字**）

| 类别 | 数量 | 用途 |
|---|---|---|
| **三模态齐全** (5mC + 5hmC + ATAC) | **2** | Fig 3-f 首选 |
| **双模态** (5mC + 5hmC，缺 ATAC) | **3** | Fig 3-f + ENCODE ATAC 补 |
| **需解压 RAW.tar** | **5** | 候选，可后续处理 |
| **单模态/部分模态** | **5** | 不推荐直接用 |
| **Mouse 已下载总计** | **15** (mouse only) + 1 (mouse+human) = **16** | |
| **真正 5mC+5hmC 可直接用的** | **5** | 立即可接入 mmllm |

## 3. mmllm 论文可直接用的 5 个 Mouse 数据集

| GSE | 5mC | 5hmC | ATAC | 测序技术 | 优先级 |
|---|---|---|---|---|---|
| **GSE166423** | ✅ direct | ✅ TAB-seq direct | ✅ broadPeak | WGBS+TAB-seq+ATAC | **P0** ⭐⭐⭐⭐⭐ |
| **GSE214845** | ✅ direct | ⚠️ BS−oxBS 推导 | ✅ bigwig | oxWGBS+ATAC | **P0** |
| **GSE244251** | ✅ Nanopore | ✅ Nanopore | ❌ (补 ENCODE) | Nanopore | **P1** ⭐⭐⭐⭐ |
| **GSE140125** | ✅ direct | ✅ TAB-seq | ❌ (补 ENCODE) | WGBS+TAB-seq | **P1** ⭐⭐⭐⭐ |
| **GSE214830** | ✅ direct | ⚠️ 推导 | ❌ (补 ENCODE) | oxWGBS | **P2** |

## 4. 需解压 RAW.tar 才能用的 5 个

| GSE | RAW 大小 | 内容 | 解压后估计可用模态 |
|---|---|---|---|
| GSE290585 | 6.5G + 1074 files | EM-seq BeadArray 534 samples | **Array 数据**（probe mapping 必需） |
| GSE186357 | 19G | 胚胎 WGBS+oxWGBS | 5mC+5hmC（早期胚胎，与 brain 不匹配） |
| GSE247534 | 1.9G | OGT mESC KO (WGBS+oxWGBS) | 5mC+5hmC（mESC KO，与你 brain 不匹配） |
| GSE141152 | 2.6G | Treg multi-omics | 5mC+5hmC+ATAC（**有 ATAC**，**解压后可用**） |
| GSE231928 | 699M | TOP-Seq 同时同 assay | 5mC+5hmC+ATAC（**有 ATAC**，**解压后可用**） |

> **GSE141152 和 GSE231928 解压后**会进入 Tier A "三模态齐全"，使总数从 2 → 4。

## 5. 不推荐直接用的（单/部分模态）

| GSE | 已有 | 缺什么 |
|---|---|---|
| GSE296587 | 5mC | 6-base CUT&Tag（count, 非 ratio；需 /coverage 处理）|
| GSE279860 | 5hmC (Nanopore bed) | 5mC 同位置 |
| GSE288331 | 5hmC + ATAC peak | 5mC |
| GSE174048 | 5mC + ATAC peak | 5hmC（在子目录，需查 GSE174047）|
| GSE150964 | ATAC peak | 5mC + 5hmC tracks 未下载 |
| GSE197740 (Mouse+Human) | 需解压 | sc，sample 数少 |

## 6. 关键结论

### 立即可接入 mmllm 的数据集数

```
Mouse + 5mC + 5hmC track-ready  →  5 个
   ├─ 三模态齐全（Fig 3-f 主选）→  2 个 (GSE166423, GSE214845)
   ├─ 双模态（需补 ENCODE ATAC）→ 3 个 (GSE244251, GSE140125, GSE214830)
```

### 解压 RAW 后可扩充至

```
Mouse + 5mC + 5hmC track-ready  →  7 个
   ├─ 三模态齐全                  →  4 个 (+ GSE141152, GSE231928)
   └─ 双模态                       →  3 个
```

### 不推荐用

- **Human 数据集**（GSE212634, GSE248149, GSE267937, GSE301936）= 4 个，需要 liftOver 到 mm10
- **Mouse 但 sample 不匹配**：GSE186357（胚胎）、GSE247534（mESC KO）
- **Array 数据**：GSE290585、GSE267937（probe mapping 必需）

## 7. Fig 3-f / Fig 4-h 推荐组合

### Fig 3-f（跨数据集泛化主图）

| 角色 | 数据集 | 备注 |
|---|---|---|
| 主候选 1 | **GSE166423** | TAB-seq direct，mm10 |
| 主候选 2 | **GSE214845** | 三模态齐全，5hmC 需 correction |
| 备用 1 | **GSE140125** | cerebellum TAB-seq |
| 备用 2 | **GSE244251** | Nanopore 黄金标准 |

### Fig 4-h（数据来源 ablation）

| 实验组 | 训练数据 |
|---|---|
| internal only | 你当前的 6 cell (AMY/HIP/PFC × MC/MW) |
| public only | GSE166423 + GSE214845 (推荐先解压 GSE141152 / GSE231928 扩到 4 个) |
| internal + public 联合 | 全部 |
| internal → public fine-tune | pretrain → fine-tune |

## 8. 优先级行动建议

1. **立即做（影响 Fig 3-f 可行性）**：
   - 解压 GSE141152 / GSE231928 的 RAW.tar → 多 2 个三模态数据集
   - 补 ENCODE forebrain/cerebellum ATAC → 多 3 个数据集支持双模态验证

2. **Fig 4-h 开始前做**：
   - 写 `correct_5hmc_negative.py` 处理 GSE214845 / GSE214830 的负值
   - 写 `normalize_chroms.py` 处理 GSE174048 的数字染色体命名
   - 写 `liftOver_human_to_mm10.py`（4 个 human 数据集如果要做迁移实验）

3. **Fig 5 阶段做**：
   - 补下 ENCODE mouse cCREs + ChromHMM（注释文件，未在 catalog 里）
   - 补下 JASPAR 2024 motif（在线数据库）

---

**结论**：可用数据够支撑 Fig 3-f / Fig 4-h 的核心叙事，**不需要再下载新数据集**。重点是把已有数据预处理打通。
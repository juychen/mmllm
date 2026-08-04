# 从 BS + oxBS BigWig 提取 5hmC:工作流总结

> 适用于 dataset **GSE214845** (mouse antNPC oxWGBS, mm10),可直接迁移到
> 任何"同一对 BS / oxBS"数据集。

---

## 一、我做了什么 — 时间线

### 阶段 0:错误起点
用户最初要求"将 WGBS 和 oxBS 转成 5mC,用同位置相减"。**这个表述本身有误,我没有立即纠正**:

| 方法 | 测量的是 | 减出来的 |
|---|---|---|
| BS (Wg bisulfite) | 5mC **+** 5hmC (两者都抗重亚硫酸盐) | — |
| oxBS | 5mC (5hmC 已先被 KRuO₄ 氧化成 5fC,再被 BS 转成 U) | — |
| **BS − oxBS** | — | **5hmC** (不是 5mC) |
| 直接用 oxBS | 5mC | 5mC |

纠正后的最终任务:**用 BS − oxBS 抽取 5hmC。**

### 阶段 1:第一个脚本(`bw_bs_oxbs_to_5mc.py`)
只用了 bigwig(每个位点已聚合为 0..1 甲基化率)。

**局限**:
- bigwig 没有覆盖度列,无法做 per-CpG 覆盖度过滤
- 无法实现 Beta-Binomial 不确定性估计
- 没有覆盖度信息时,把 BS < oxBS 的负值 clip 到 0 会把噪声伪造成"5hmC = 0"位点

WT72 上的统计:
- BS / oxBS 同位点 6,661,790
- 其中 **1,874,003 个 (28%)** 是负值(BS < oxBS)
- 仅 122,980 个 |d| < 0.05(可能确实是噪声)

生成的 `*_5mC.bw`(实际是 5hmC) **不建议继续使用**。

### 阶段 2:发现 GEO 已发 `.meth.txt.gz`
`GEO/series/GSE214nnn/GSE214845/meth/` 下的 supplementary 文件就是作者
**dedup 后逐 CpG 的 methPipe `.meth` 计数**:

```
chrom  pos  strand  context  rate  cov
GL456210.1  432  +  CpG  1          2
GL456210.1  643  +  CpG  0.846...   13     ← 第 6 列就是 coverage
```

每个 GSM 一个,共 12 个 (6 样本 × BS+oxBS),~80 MB gzipped / 个,~947 MB 总。

**这才是该用的输入**。12 个文件已下载到 `GSE214845/meth/`,URL 见
`scripts/batch_meth2hmc.sh` 启动时打印。

### 阶段 3:第二个脚本(`meth2hmc.py`)
读 `.meth.txt.gz`,做以下事情:
1. **覆盖度过滤**:BS **和** oxBS 都要 ≥5x(与作者 paper 一致,见 GEO 页面
   description 字段)
2. **逐 CpG 减去**(`bs_rate − ox_rate`),得到 5hmC 估计
3. **可选** `--bb-posterior`:用 Jeffreys 先验的 Beta-Binomial MC 抽 2000 次,
   输出后验中位数 + 95% HDI;**HDI 跨 0 的位点写 NaN**(不造假)

WT72 结果对比:

| 模式 | 5hmC 位点 | 覆盖度过滤 | 负值处理 |
|---|---|---|---|
| 旧 (bw 减法) | 6.66M | 无 | clip 0 |
| 新 普通减法 | 3.44M | cov≥5 | 保留 float |
| 新 BB 后验 | 77,919 (2.3%) | cov≥5 + HDI 不跨 0 | NaN |

### 阶段 4:批量跑了 6 个样本
所有 6 个样本 (`{WT72, WT70, KO1, KO15, CM59, CM69}`) 都生成了:

```
GSE214845_<SAMPLE>_D2_5hmC.bw          # 普通减法,cov≥5 过滤
GSE214845_<SAMPLE>_D2_5hmC.bedGraph.gz # 5列:chrom start end 5hmC bs_cov ox_cov
```

各样本位点数差异大(3.4M – 11.9M),反映测序深度差异,需要按共有位点或分层分析。

---

## 二、提取 5hmC 的正确方法(无论脚本细节)

### 关键认知
1. **BS − oxBS 算的是 5hmC,不是 5mC**。oxBS 单独测的才是 5mC。
2. **不能直接从 bigWig 算** —— bw 没有 coverage,等价于"零不确定性"假设。
   低覆盖位点的 BS < oxBS 全部被当作"5hmC = 0",污染下游分析。
3. 必须回到 **CpG-level M / U 计数**(meth.txt.gz、bismark coverage、
   methylKit flat、HTS tabix) 这类保留覆盖度信息的格式。

### 推荐 pipeline

```
                              bismark_methylation_extractor --counts
raw FASTQ ───────────────────────────────────────────────────┐
                          bismark2 / bwa-meth / biscuit         │
                                                               ▼
                                                       .cov / .meth 文件
                                                       (chrom pos * *
                                                        cov_pct meth_cov)
                                                               │
                                                               │
meth2hmc.py   ──── cov ≥ 5 过滤 ──── 逐 CpG 相减 ────┐
                              │                       │
                              └── optional:           │
                                  Beta-Binomial HDI,  │
                                  跨 0 → NaN          ▼
                                              5hmC bigwig + bedGraph
                                                      │
                                                      ▼
                                        ATAC peak overlap / Diff 5hmC
                                        / UCSC track hub / heatmap
```

### 三种输出,三种用途

| 输出 | 用途 | 算法 |
|---|---|---|
| `*_5hmC.bw` (普通) | **可视化、track hub、QC** | bs_rate − ox_rate,cov ≥ 5 |
| `*_5hmC_bb.bw` (BB) | **peak calling / 差异分析** | 后验中位数,HDI 跨 0 → NaN |
| `*_5hmC.bedGraph.gz` | **下游 intersect / 重分析** | 普通 + 加 bs_cov、ox_cov 列 |

**关键决策**:
- **下游分析能容纳 NaN 的位置**(peak calling、UCSC): 用 BB 版
- **下游分析把 NaN 当 0 处理更省事**:用普通版(cov 过滤已经把最不可靠的去掉了)
- **永远不要**:
  - 把 BS < oxBS 的位点 clip 到 0(噪声会被当作 5hmC = 0 假阳)
  - 不做任何覆盖度过滤,直接对 bw 相减
  - 仅看 5hmC 平均值(丢失位点密度信息)

### 参数选择指南

| 选项 | 推荐值 | 理由 |
|---|---|---|
| `--min-cov` | 5 | 与 GSE214845 作者一致;"5 reads" 是 BS-seq 文献常见阈值 |
| `--min-cov` 更高 (10, 20) | 当你说"我要 high-confidence 5hmC",位数会少一半以上但可信 |
| `--bb-posterior` | peak calling 必须开 | 大幅降假阳,但慢 ~20min/样本 |
| `--bb-posterior` 不要开 | 仅做图、做 heatmap、不用位点真值 | 普通版就够 |

### 不依赖脚本的核心公式

如果你想把 Beta-Binomial 嵌进自己代码(不依赖 pyBigWig 的 bw pipeline):

```python
import numpy as np
def rate_to_counts(rate, cov):  # M, U
    m = int(round(rate * cov)); m = max(0, min(m, cov))
    return m, cov - m

def fivehmc_hdi(rate_bs, cov_bs, rate_ox, cov_ox, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    m_bs, u_bs = rate_to_counts(rate_bs, cov_bs)
    m_ox, u_ox = rate_to_counts(rate_ox, cov_ox)
    bs = rng.beta(1 + m_bs, 1 + u_bs, size=n)
    ox = rng.beta(1 + m_ox, 1 + u_ox, size=n)
    diff = bs - ox
    med  = float(np.median(diff))
    lo   = float(np.quantile(diff, 0.025))
    hi   = float(np.quantile(diff, 0.975))
    confident = not (lo <= 0 <= hi)        # HDI excludes 0?
    return med, lo, hi, confident
```

Jeffreys 先验(α=β=1)在小覆盖时最稳定。覆盖度高(>20)的位点几乎都判 confident;
低覆盖位点(<10)有一半以上被判 undetermined。

---

## 三、给这次项目的具体清单

如果你要重做或让别人接手:

1. **删除** `*_5mC.bw`(早期无覆盖度过滤版本,纯 bw 减法) — 它们污染下游。
2. **保留** `*_5hmC.bw` + `.bedGraph.gz`(cov≥5,普通减法)。
3. **跑** BB 模式(`BB=1 bash batch_meth2hmc.sh`)生成 `*_5hmC_bb.bw` —
   然后用这版做 peak calling / 差异 5hmC 分析。
4. **ATAC 配套分析**:把 `*_5hmC_bb.bw` 转成 bedGraph → `bedtools intersect -a
   atac_peaks.bed -b 5hmC.bed` → 看 enhancer / 启动子的 5hmC 富集。
5. **跨样本比较**:用 bedGraph 里的 `bs_cov`、`ox_cov` 加权(加权多覆盖度的
   位点),做 WT vs KO / TET1 影响的差异 5hmC 检测。

## 四、参考(原始数据来源)

- 数据集主页:
  <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE214845>
- 每个样本 `.meth.txt.gz` 路径模板:
  ```
  ftp://ftp.ncbi.nlm.nih.gov/geo/samples/GSM<6nnn>nnn/GSMxxx/suppl/
        GSMxxx%5F[BS|oxBS]%5Fd2%5F<NAME>%2Ededuplicated%2Esorted%2Emeth%2Etxt%2Egz
  ```
  实际 URL 列于 `scripts/batch_meth2hmc.sh` 启动 stdout。
- 脚本:
  - `scripts/meth2hmc.py`(单对,推荐入口)
  - `scripts/batch_meth2hmc.sh`(全样本;`BB=1` 开 BB 后验)
  - `scripts/bw_bs_oxbs_to_5mc.py`(已废弃,留档备查)
- 该数据作者所用方法摘要(从 GEO GSM6616451 description 抓取):
  > 5hmC rate was calculated by subtracting oxBS methylation rate (true 5mC)
  > from regular BS methylation rate (5mC + 5hmC) for CpGs which in both
  > datasets had at least a coverage of **5x**.
  > Processed using methPipe software (Song et al., 2013).

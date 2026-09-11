# zcodeplanformultimodal —— 下游训练防遗忘方案（方案二：per-dataset Adapter + 冻结主干）

> 状态：**已定稿，尚未实施**。本文档只记录方案，不含任何代码改动。

## 0. 背景与决策记录

**起因**：已加入"模态级 20% dropout"（整条 context 模态以 20% 概率被 `[MASK]` 替换，即 `--seq-drop-p / --atac-drop-p / --rna-drop-p`）。目标：主模型训练完后，接入公共数据集训练时允许模态缺失，且**原域（AMY/HIP/PFC × MC/MW）指标不能变差**。

**三个候选方案对比**

| 维度 | 一 Replay 混合联合 | 二 Adapter + 冻结主干（选定） | 三 锚定正则 + 权重融合 |
|---|---|---|---|
| 作用层 | 数据 | 结构 | 优化/后处理 |
| 原域保证 | 概率（可调） | **结构保证（零退化）** | 概率（可调） |
| 下游上限 | **最高** | 中 | 中~高 |
| 抗技术差异 | 弱 | **最强（参数隔离）** | 中 |
| 保留缺模态鲁棒性 | 会被侵蚀 | **完整保留** | 较好 |
| 代码改动 | 中（须修 lazy 路径） | 中（加 adapter） | **最小** |
| 前置依赖 | 修多数据集 lazy 读错 track | 无（单数据集/run） | (b) 无条件 |

**选择理由**：硬要求是"原域不降"，只有方案二能给出结构性保证（主干不动、原域走原路径、adapter 零初始化即 identity），且天然保留 dropout 学到的缺模态鲁棒性。方案三作为后续加固，方案一作为"需要把公共数据学进共享权重"时的可选增强。

**关键代码事实（已核实）**

- mask token 只存在于 RNA 版 `M5CQuerySequenceAtacRnaCrossHyenaRegressorModelB`（`models.py:520-554`）；非 RNA 版与 baseline 无法表达模态缺失。
- RNA 缺失必须表达为"零张量 + `rna_present=False`"；`rna_track=None` 且 `rna_proj` 存在会 shape mismatch 崩溃。
- 训练 dropout 是**每 batch 抽一次**（整批共享），非 per-sample；`evaluate()` 硬编码全模态 → 缺模态鲁棒性目前**零证据**。
- 多数据集 lazy 路径**读死 `paths[0]`**（`data.py:1021-1062` 只用构造函数句柄，忽略 per-row `*_path`）→ 混合多组会静默读错 track。（方案二单数据集/run，不触发。）
- 5mC/5hmC 走**原始单位且无归一化**，不同生成脚本产出 0–100 与 0–1 两种尺度；ATAC/RNA 是 per-window minmax → 跨数据集尺度不一致是"技术差异搞坏模型"的头号原因。
- 现有钩子：`--init-from-checkpoint`（只载权重）、`--freeze-backbone-epochs`（冻结除 head 外全部）、patience 早停、best 按 val_loss 选；**无梯度裁剪、无 per-epoch checkpoint**。

## 1. 核心设计

- 主干 = 已训练好的 5mC→5hmC RNA 版模型，**全程冻结**。
- 每数据集一个 `DomainAdapter`（bottleneck 残差 MLP，最后一层 zero-init → 初始严格 identity）。
- 原域推理**不带 adapter** → 指标与训练前逐位一致（结构保证）。
- **只训 adapter**，head 一并冻结；`--unfreeze-head` 作为容量不足时的退路（会打破严格保证，需重新测量原域）。

## 2. 文件级改动

### 2.1 `models.py`

- 新增 `DomainAdapter(nn.Module)`：
  `x + Linear(adapter_dim→hidden)( SiLU( Linear(hidden→adapter_dim)( LayerNorm(x) ) ) )`，最后一层 weight/bias **zero-init**（与 `CrossAttentionResidualBranch` 同款做法）。
- `M5CQuerySequenceAtacRnaCrossHyenaRegressorModelB.__init__` 新增：
  `adapter_names: Sequence[str] = ()`、`adapter_dim: int | None = None`、`adapter_points: str = "context"`；
  构建 `self.context_adapters = nn.ModuleDict(...)`（`adapter_points` 含 hidden 时再建 `self.hidden_adapters`）。
- `forward(..., domain_id: str | None = None)`：
  context adapter 插在 `context_norm(context_proj(...))` 之后；hidden adapter 插在 `final_norm` 之前。
  `domain_id=None` 或未注册 → 完全跳过，**与现行为 bit-identical**。
- 兼容性：adapter 是新增参数、不改任何既有层形状 → 旧 checkpoint 经 `load_state_dict(strict=False)` 干净加载（现有 `load_model_from_checkpoint` 已走 non-strict 分支）。
- （可选后续）非 RNA 版 `M5CQuerySequenceAtacCrossHyenaRegressorModelB` 同样处理。

### 2.2 `data.py`（最小改动）

- 给 `LazyM5cSequenceAtacDataset` / `LazyM5cSequenceAtacRnaDataset.__init__` 增加 per-modality 尺度参数
  `m5c_scale / hm5c_scale / atac_scale / rna_scale`（默认 1.0），在 `__getitem__` 读取后相乘；
  用于把公共数据集的 0–100 或 0–1 换算到主干训练尺度（主干 5mC/5hmC 是原始单位；ATAC/RNA 已 per-window minmax）。
- 加运行时值域检查：某模态 |value| 上界 >2 且对应 scale 仍为 1.0 → 打印告警并写入结果 JSON（防止静默混尺度）。
- **不做** per-row 路径改造（本方案不需要）。

### 2.3 新脚本 `run_adapter_finetune.py`（现有训练脚本一律不动）

- import 复用现有 helper：`prepare_sequence_atac_crosshyena_data`、`masked_mse_loss`、`evaluate`、
  `collect_predictions`、`load_model_from_checkpoint`、`build_optimizer`、`build_scheduler`、`save_checkpoint`。
- 新增参数：
  `--backbone-checkpoint`、`--adapter-name`、`--adapter-dim`、`--adapter-points {context,hidden,both}`、
  `--unfreeze-head`、`--grad-clip-norm 1.0`、`--learning-rate 3e-4`、`--epochs`、`--patience`、
  `--dataset-name`、`--eval-modality-matrix`、`--eval-original-val <csv>`。
- 流程：构造带 adapter 名的 RNA 模型 → 载主干 → 冻结全部参数 → 解冻 `adapters.<name>.*`（可选 `head.*`）
  → `build_optimizer`（已按 `requires_grad` 过滤）→ 训练（梯度裁剪 + AMP）→ 评估（下游 / 原域 / 模态矩阵）
  → 存 `adapter_<dataset>.pt`（**独立文件，不覆盖原 checkpoint**）。
- 自带 `checkpointed_forward`（含 adapter 应用），兼容 `--gradient-checkpointing`。

### 2.4 推理与上报

- `run_m5c_inference.py` 增加 `--adapter-name`（可选，默认无 → 原域零退化）。
- 结果 JSON 扩展字段：
  `dataset_name`、`backbone_checkpoint`、`trainable_param_count`、`frozen_param_count`、
  `per_modality_combo`、`original_val_before/after`。

## 3. 验收测试（按序）

1. **T1 结构保证**：同一 checkpoint，原域 val 在「带 adapter」与「不带 adapter」两种推理下指标**完全相同**。
2. **T2 zero-init 自检**：adapter 权重全零时，输出与不带 adapter 一致。
3. **T3 缺模态矩阵**：全模态 vs 各缺失组合的 pearson/r2/loss —— 这是"模态可以缺失"的**唯一证据**，必须先跑出来再依赖。
4. **T4 端到端**：选一个公共数据集（建议 GSE166423 Purkinje Adult，或 GSE244251 细胞类型）训 adapter
   → 下游指标优于 zero-shot，且原域 before/after 差为 0。

## 4. 不做

- 不改 pretraining、不改现有 dropmod 训练脚本、不修多数据集 lazy 路径（本方案不需要）
- 不引入 DANN/CORAL 对抗域适应
- 不实现 WiSE-FT / 锚定正则（方案三，留作后续加固）

## 5. 成本与风险

- 成本：`models.py` ~2h；`data.py` 尺度 ~1h；新脚本 ~半天；评测/验收 ~2h。合计约 1 天。
- 风险：
  1. adapter-only 容量可能不足以覆盖大域移（跨物种/跨 assay）→ 退路是 `--unfreeze-head` 或解冻最后一个 block，代价是原域不再严格零退化，需重新测量并写入实验表；
  2. 5mC/5hmC 尺度不统一必须显式配 scale（或不统一就换 track）；
  3. 缺模态鲁棒性目前是设计意图、零实测证据，T3 必须先于任何结论。

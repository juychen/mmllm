# Reference-Anchored Cell-Type Methylation Mixture Model
## Coding specification for bulk Nanopore methylation deconvolution with incomplete cell-type references

**Status:** implementation specification  
**Primary goal:** infer cell-type-resolved disease-associated methylation perturbations from bulk control/disease methylation tracks, using control-only cell-type methylation references and single-cell-derived cell-composition priors.

---

# 1. Scientific objective

We have:

1. **Control-only cell-type-specific methylation reference tracks**
   - e.g. 4 purified/sorted cell types measured by third-generation sequencing.
   - each track contains methylation percentages at CpG positions or aligned methylation loci.
   - reference cell types are **not necessarily exhaustive**.

2. **Our own bulk control samples**
   - methylation track per sample.

3. **Our own bulk disease/experimental samples**
   - methylation track per sample.
   - control-vs-disease DMRs are already available.

4. **Single-cell RNA/ATAC data**
   - used to derive priors for cell-type proportions.
   - may contain more cell types than are available in the methylation reference.

The model must satisfy two core generative constraints:

\[
\hat{\mathbf y}^{CTRL}_s
=
\sum_{c} \pi^{CTRL}_{s,c}\mathbf\theta^{0}_{c}
\approx
\mathbf y^{CTRL}_s
\]

and

\[
\hat{\mathbf y}^{DIS}_s
=
\sum_{c} \pi^{DIS}_{s,c}
\mathbf\theta^{DIS}_{s,c}
\approx
\mathbf y^{DIS}_s
\]

where

\[
\mathbf\theta^{DIS}_{s,c}
=
\sigma\left(
\operatorname{logit}(\mathbf\theta^{0}_{c})
+
\mathbf\Delta_{s,c}
\right)
\]

The model therefore decomposes observed bulk disease methylation into:

- **cell-composition changes**
- **cell-intrinsic methylation changes**
- an explicit **OTHER/unreferenced cell compartment**, when the reference does not cover all cell types.

The model must **not** use an unrestricted post-mixture disease residual in the primary implementation, because such a residual would make the cell-type attribution non-identifiable. Missing biology must instead be absorbed by the explicit OTHER component or reported as reconstruction failure.

---

# 2. Core design principle for incomplete reference cell types

Let the single-cell data contain cell types:

\[
\mathcal C_{all}
\]

and methylation references exist for:

\[
\mathcal C_{ref} \subset \mathcal C_{all}
\]

All single-cell cell types without methylation references are aggregated into one latent component:

\[
c = OTHER
\]

Therefore the mixture components are:

\[
\mathcal C =
\mathcal C_{ref} \cup \{OTHER\}
\]

Example:

```text
Single-cell composition:
Glut        0.52
GABA        0.18
Astro       0.12
Oligo       0.10
OPC         0.03
Microglia   0.03
Endothelial 0.02

Available methylation references:
Glut
GABA
Astro
Oligo

Model prior:
Glut   0.52
GABA   0.18
Astro  0.12
Oligo  0.10
OTHER  0.08
```

The first implementation MUST use a **single aggregated OTHER component** rather than multiple unconstrained missing cell types.

Reason: bulk methylation alone generally cannot distinguish several unreferenced cell types without additional cell-type-specific methylation references.

---

# 3. Required input files

## 3.1 Reference methylation tracks

Required columns:

```text
locus_id
chrom
start
end
cell_type
meth_value
valid
```

Optional but strongly recommended:

```text
coverage
modified_reads
total_reads
replicate_id
```

Requirements:

- `meth_value` must be converted to `[0, 1]`.
- missing observations must be represented by `valid = 0`, not by `meth_value = 0`.
- exact `0` and `1` are valid methylation proportions.

## 3.2 Bulk control/disease methylation tracks

Required:

```text
sample_id
condition
locus_id
meth_value
valid
```

Recommended:

```text
coverage
modified_reads
total_reads
brain_region
sex
batch
```

`condition` must be:

```text
CTRL
DIS
```

## 3.3 Single-cell composition prior

Required:

```text
sample_id
cell_type
prior_fraction
```

If single-cell data are not matched to the exact bulk sample, allow:

```text
group_id
brain_region
condition
cell_type
prior_fraction
```

Then map each bulk sample to the appropriate prior group.

For cell types without methylation references:

```text
prior_OTHER =
sum(prior_fraction of all unreferenced cell types)
```

## 3.4 DMR annotation

Required:

```text
locus_id
is_dmr
dmr_id
```

Optional:

```text
delta_bulk
fdr
direction
```

For version 1, disease perturbation should be allowed primarily on DMR loci.

## 3.5 Missing signal and locus/region masking

`valid = 0` marks a missing observation and is never turned into a value of `0`
(criterion 7). Where a track has `coverage`, `coverage <= 0` counts as no signal
as well, as does `coverage < min_coverage` when a threshold is configured. The
same rule applies to reference and bulk tracks.

Per-locus reference mask:

- Build a mask `ref_mask[cell_type, locus]`, True where that reference has a
  signal at that locus.
- **Partial missingness needs no special treatment.** As long as at least one
  reference observes the locus, the locus is kept and fits normally. Entries the
  mask flags False keep an imputed placeholder (the locus mean of the observed
  references, falling back to the global mean) purely so the reference matrix
  stays finite; they carry no information.
- A delta on a masked `(cell_type, locus)` entry is forced to `0`: no data
  supports it, so it must not enter the L1 / group / smooth penalties nor the
  attribution table.

Locus-level exclusion:

- A locus no reference observes carries no information about the mixture. It
  keeps its row (so the output tables stay aligned) but is excluded from the
  reconstruction loss, together with the delta penalties.
- The effective reconstruction mask is therefore
  `valid AND weight > 0 AND (some reference observed at the locus)`. Entries
  excluded this way contribute to neither the numerator nor the denominator of
  `L_recon`, and the QC metrics use the same rule so they describe the population
  the loss actually optimises.

Region-level exclusion:

- A DMR whose **every** locus is missing in **every** reference is dropped as a
  unit: `is_dmr = 0` for its loci, so its delta stays out of the penalties and the
  attribution, and the dropped `dmr_id`s are reported.

Bulk observations keep their own `mask[sample, locus]`; the two masks compose.

Per-locus output tables (`celltype_baseline_methylation.tsv.gz`,
`celltype_disease_delta.tsv.gz`, `bulk_reconstruction.tsv.gz`) cost one row per
locus per component or sample. At genome-wide locus counts they are not
affordable, so they are written only when the locus count is below the configured
limit; the aggregated tables (`reference_missingness.tsv.gz`, `dropped_dmrs.tsv`,
`dmr_attribution.tsv.gz`) are always written.

---

# 4. Reference methylation distribution

Each reference cell type is represented as a probability distribution rather than a fixed noiseless value.

For cell type \(c\), locus \(r\):

\[
\theta^0_{c,r}
\sim
Beta(\alpha_{c,r}, \beta_{c,r})
\]

with mean:

\[
\mu^0_{c,r}
=
\frac{\alpha_{c,r}}
{\alpha_{c,r}+\beta_{c,r}}
\]

If modified/total read counts are available:

\[
\alpha_{c,r}=K_{c,r}+a_0
\]

\[
\beta_{c,r}=N_{c,r}-K_{c,r}+b_0
\]

Default:

```yaml
beta_prior_a0: 0.5
beta_prior_b0: 0.5
```

If only methylation percentages are available:

\[
\alpha_{c,r}
=
\mu^0_{c,r}\kappa_{ref}
\]

\[
\beta_{c,r}
=
(1-\mu^0_{c,r})\kappa_{ref}
\]

Default:

```yaml
reference_concentration: 50.0
```

---

# 5. Cell-proportion prior

Cell proportions are latent simplex variables:

\[
\pi_{s,c} \ge 0
\]

\[
\sum_c \pi_{s,c} = 1
\]

The single-cell composition provides the prior center:

\[
p^{SC}_{s,c}
\]

Use a Dirichlet prior:

\[
\pi_s
\sim
Dirichlet(
\kappa_\pi p^{SC}_{s}
+
\epsilon
)
\]

Defaults:

```yaml
composition_prior_strength: 100.0
composition_epsilon: 1.0e-4
```

Important:

- CTRL and DIS samples may have different priors.
- DO NOT force `pi_CTRL = pi_DIS`.
- if matched single-cell proportions are reliable, provide an option to fix `pi_s = p_SC_s` rather than learn them.

Modes:

```yaml
composition_mode: fixed
# or
composition_mode: dirichlet_prior
```

---

# 6. The OTHER component

For each sample:

\[
p^{SC}_{s,OTHER}
=
1
-
\sum_{c\in\mathcal C_{ref}}
p^{SC}_{s,c}
\]

The OTHER control baseline is unknown:

\[
\theta^0_{OTHER,r}
\]

Initialize it from the control residual when possible:

\[
\theta^{init}_{OTHER,r}
=
\operatorname{clip}
\left(
\frac{
\bar y^{CTRL}_{r}
-
\sum_{c\in\mathcal C_{ref}}
\bar\pi^{CTRL}_{c}\theta^0_{c,r}
}{
\bar\pi^{CTRL}_{OTHER}
},
\epsilon,
1-\epsilon
\right)
\]

Regularize:

\[
L_{other-anchor}
=
\lambda_{other}
\left\|
logit(\theta^0_{OTHER})
-
logit(\theta^{init}_{OTHER})
\right\|^2
\]

The OTHER component must not be a completely free track.

---

# 7. Control generative model

For control sample \(s\), locus \(r\):

\[
\boxed{
\hat y^{CTRL}_{s,r}
=
\sum_{c\in\mathcal C}
\pi^{CTRL}_{s,c}
\theta^0_{c,r}
}
\]

This is a REQUIRED architectural identity.

No arbitrary control residual branch is allowed in the primary model.

---

# 8. Disease generative model

For disease sample \(s\), cell type \(c\), locus \(r\):

\[
\eta^0_{c,r}
=
logit(\theta^0_{c,r})
\]

\[
\theta^{DIS}_{s,c,r}
=
\sigma(
\eta^0_{c,r}
+
\Delta_{s,c,r}
)
\]

Then:

\[
\boxed{
\hat y^{DIS}_{s,r}
=
\sum_{c\in\mathcal C}
\pi^{DIS}_{s,c}
\theta^{DIS}_{s,c,r}
}
\]

This is the second REQUIRED architectural identity.

The disease effect must occur **before mixture**.

Do NOT implement:

\[
\hat y^{DIS}
=
\sum_c\pi_c\theta^0_c
+
\Delta_{bulk}
\]

as the primary model.

---

# 9. Disease perturbation parameterization

The naive tensor

\[
\Delta_{s,c,r}
\]

is highly underdetermined.

## Version 1: group-level sparse delta

Use:

\[
\Delta_{c,r}
\]

shared across disease samples within the same analysis group.

For control samples:

\[
\Delta_{c,r}=0
\]

For disease samples:

\[
\theta^{DIS}_{c,r}
=
\sigma(
logit(\theta^0_{c,r})
+
\Delta_{c,r}
)
\]

Default:

```yaml
delta_scope: dmr_only
```

Use L1 sparsity:

\[
L_{\Delta,L1}
=
\lambda_{\Delta}
\sum_{c,r}
|\Delta_{c,r}|
\]

Optional group sparsity:

\[
L_{\Delta,group}
=
\lambda_{group}
\sum_r
\|\Delta_{\cdot,r}\|_2
\]

---

# 10. VAE extension

After version 1 is stable, implement a scVI-like latent sample state.

Prior:

\[
z_s \sim N(0,I)
\]

Encoder:

\[
q_\phi(z_s|\mathbf y_s, \mathbf m_s, condition_s)
=
N(\mu_s,\sigma_s^2)
\]

Decoder:

\[
\Delta_{s,c,r}
=
g_\theta(z_s,e_c,h_r)
\]

Cell-type methylation:

\[
\theta_{s,c,r}
=
\sigma(
logit(\theta^0_{c,r})
+
d_s\Delta_{s,c,r}
)
\]

where:

\[
d_s =
\begin{cases}
0 & CTRL\\
1 & DIS
\end{cases}
\]

Mixture:

\[
\hat y_{s,r}
=
\sum_c
\pi_{s,c}\theta_{s,c,r}
\]

VAE objective adds:

\[
L_{KL,z}
=
KL(
q_\phi(z_s|\cdot)
\|
N(0,I)
)
\]

Use KL warmup.

---

# 11. Observation model for percentage tracks

All methylation values are continuous percentages converted to `[0,1]`.

Default implementation: masked Huber loss.

\[
L_{recon}
=
\frac{
\sum_{s,r}
m_{s,r}
w_{s,r}
Huber(
y_{s,r},
\hat y_{s,r}
)
}{
\sum_{s,r}
m_{s,r}w_{s,r}
}
\]

where:

- `m`: valid observation mask
- `w`: optional coverage weight

If counts are available, optionally use Binomial or Beta-Binomial likelihood.

---

# 12. Required losses

\[
L
=
\lambda_C L_C
+
\lambda_D L_D
+
\lambda_\pi L_\pi
+
\lambda_\Delta L_{\Delta,L1}
+
\lambda_G L_{\Delta,group}
+
\lambda_O L_{other-anchor}
+
\lambda_S L_{smooth}
\]

with:

\[
L_C
=
Recon(
\mathbf y^{CTRL},
\hat{\mathbf y}^{CTRL}
)
\]

\[
L_D
=
Recon(
\mathbf y^{DIS},
\hat{\mathbf y}^{DIS}
)
\]

If composition is learned:

\[
L_\pi
=
-
\log p(
\pi_s |
Dirichlet(
\kappa_\pi p^{SC}_s
)
)
\]

---

# 13. Near-hard reconstruction constraint

The model must make:

1. reference mixture reconstruct the user's CTRL bulk
2. reference + delta mixture reconstruct the user's DIS bulk

Exact equality is not always feasible because of noise and incomplete references.

Support two modes:

## Standard soft constraint

```yaml
lambda_control_recon: 10.0
lambda_disease_recon: 10.0
```

## Optional augmented-Lagrangian mode

For residuals:

\[
r_C = \hat y_C-y_C
\]

\[
r_D = \hat y_D-y_D
\]

add:

\[
L_{AL}
=
\lambda_C^T r_C
+
\frac{\rho_C}{2}\|r_C\|^2
+
\lambda_D^T r_D
+
\frac{\rho_D}{2}\|r_D\|^2
\]

---

# 14. Bulk disease-effect decomposition

For locus \(r\):

\[
\Delta^{bulk}_r
=
\hat y^{DIS}_r
-
\hat y^{CTRL}_r
\]

Composition effect:

\[
\Delta^{comp}_r
=
\sum_c
(\bar\pi^D_c-\bar\pi^C_c)
\theta^0_{c,r}
\]

Cell-intrinsic disease effect:

\[
\Delta^{intrinsic}_r
=
\sum_c
\bar\pi^D_c
(
\theta^{DIS}_{c,r}
-
\theta^0_{c,r}
)
\]

Per-cell-type contribution:

\[
Contribution_{c,r}
=
\bar\pi^D_c
(
\theta^{DIS}_{c,r}
-
\theta^0_{c,r}
)
\]

---

# 15. Required outputs

## `sample_composition_posterior.tsv`

```text
sample_id
condition
cell_type
prior_fraction
posterior_fraction
```

## `celltype_baseline_methylation.tsv.gz`

```text
locus_id
cell_type
theta0_mean
theta0_sd
is_reference
```

## `celltype_disease_delta.tsv.gz`

```text
locus_id
dmr_id
cell_type
delta_logit
theta_ctrl
theta_disease
delta_probability
```

## `bulk_reconstruction.tsv.gz`

```text
sample_id
condition
locus_id
observed
predicted
residual
valid
```

## `dmr_attribution.tsv.gz`

```text
dmr_id
cell_type
composition_effect
intrinsic_effect
celltype_intrinsic_contribution
abs_contribution_fraction
```

## `model_qc.json`

Include:

```text
control_rmse
control_mae
control_pearson
disease_rmse
disease_mae
disease_pearson
fraction_control_within_tolerance
fraction_disease_within_tolerance
composition_prior_deviation
other_fraction_summary
delta_sparsity
```

---

# 16. Required QC and identifiability checks

1. Control reconstruction must pass before disease attribution is interpreted.
2. Disease reconstruction must pass after delta fitting.
3. Prior sensitivity:
   - `kappa_pi = 20, 50, 100, 200`
4. OTHER sensitivity:
   - run with OTHER enabled and disabled.
5. Delta sparsity sensitivity.
6. Leave-one-reference-cell-type-out test:
   - hide one known reference
   - move its single-cell fraction into OTHER
   - refit
   - quantify reconstruction degradation.
7. Synthetic mixture benchmark:
   - generate known mixtures from reference tracks
   - inject known cell-type-specific deltas
   - test recovery of delta sign, magnitude and responsible cell type.

---

# 17. Train/validation/test split

Do NOT randomly split overlapping genomic windows.

Prefer chromosome/block split.

Example:

```yaml
train_chromosomes:
  - chr1
  - chr2
  - chr3
  - chr4
  - chr5
  - chr6
  - chr7
  - chr8
  - chr9
  - chr10
  - chr11
  - chr12
  - chr13
  - chr14
  - chr15

validation_chromosomes:
  - chr16
  - chr17

test_chromosomes:
  - chr18
  - chr19
```

---

# 18. Recommended training stages

## Stage 0 — data alignment

- intersect loci across reference, CTRL and DIS tracks
- preserve masks
- convert percentages to `[0,1]`
- aggregate single-cell priors into known references + OTHER
- verify all proportions sum to 1

## Stage 1 — reference distribution fitting

Estimate Beta distributions for known references.

Initialize OTHER.

## Stage 2 — control-only mixture fitting

Fit:

- `pi_CTRL` if not fixed
- `theta0_OTHER`

Goal:

\[
\hat y^{CTRL}
\approx
y^{CTRL}
\]

Do not continue if control QC fails.

## Stage 3 — disease mixture fitting

Fit:

- `pi_DIS` if not fixed
- `Delta[c,r]` on DMR loci
- optional `Delta_OTHER,r`, strongly regularized

Goal:

\[
\hat y^{DIS}
\approx
y^{DIS}
\]

## Stage 4 — attribution

Compute composition and intrinsic effects.

## Stage 5 — optional VAE

Only after deterministic model passes all QC.

---

# 19. PyTorch project layout

```text
methylmix/
├── README.md
├── configs/
│   └── default.yaml
├── src/
│   ├── data.py
│   ├── reference.py
│   ├── composition.py
│   ├── mixture_model.py
│   ├── likelihoods.py
│   ├── losses.py
│   ├── train_stage1_reference.py
│   ├── train_stage2_control.py
│   ├── train_stage3_disease.py
│   ├── attribution.py
│   ├── qc.py
│   └── utils.py
├── scripts/
│   ├── run_all.sh
│   ├── resume.sh
│   └── synthetic_benchmark.sh
├── tests/
│   ├── test_mixture_identity.py
│   ├── test_simplex.py
│   ├── test_other_component.py
│   ├── test_delta_bounds.py
│   └── test_synthetic_recovery.py
└── outputs/
```

---

# 20. Core PyTorch parameterization

Composition:

```python
pi_logits = nn.Parameter(torch.zeros(n_samples, n_components))
pi = torch.softmax(pi_logits, dim=-1)
```

Known reference baseline:

```python
self.register_buffer("theta_ref", theta_ref)
```

OTHER baseline:

```python
other_logits = nn.Parameter(logit(other_init))
theta_other = torch.sigmoid(other_logits)
```

Disease delta:

```python
delta = nn.Parameter(torch.zeros(n_components, n_loci))
delta_eff = delta * dmr_mask[None, :]
```

Disease cell-type track:

```python
theta_dis = torch.sigmoid(
    torch.logit(theta0.clamp(eps, 1-eps))
    + delta_eff
)
```

Control mixture:

```python
pred_ctrl = torch.einsum(
    "sc,cr->sr",
    pi_ctrl,
    theta0
)
```

Disease mixture:

```python
pred_dis = torch.einsum(
    "sc,scr->sr",
    pi_dis,
    theta_dis
)
```

---

# 21. Mandatory unit tests

## Test 1: mixture identity

Given:

```text
theta:
cell1 = [0.8, 0.2]
cell2 = [0.2, 0.6]

pi = [0.75, 0.25]
```

Expected:

```text
bulk = [0.65, 0.30]
```

## Test 2: simplex

```python
assert torch.all(pi >= 0)
assert torch.allclose(
    pi.sum(dim=-1),
    torch.ones_like(pi.sum(dim=-1)),
    atol=1e-6
)
```

## Test 3: methylation bounds

Require:

```python
0 <= theta0 <= 1
0 <= theta_dis <= 1
0 <= pred_ctrl <= 1
0 <= pred_dis <= 1
```

## Test 4: delta = 0

When:

```python
delta[:] = 0
pi_dis = pi_ctrl
```

require:

```python
pred_dis == pred_ctrl
```

## Test 5: missing reference

Remove one known cell type, aggregate its fraction into OTHER, and verify the model still runs.

## Test 6: synthetic delta recovery

Inject delta into one cell type and a subset of loci; the inferred top cell type should match the simulated truth.

## Test 7: masked bulk entry is inert

Mask one bulk observation. A non-finite placeholder and a wild finite value at
that entry must both leave `L` exactly as the mask left it.

## Test 8: masked reference entry carries no delta

Mask one `(cell_type, locus)` entry. `delta_eff` must be `0` there, and changing
`delta` at that entry must not move `L`. The OTHER row is never masked.

## Test 9: locus no reference observes is excluded

Mask every reference at one locus. The bulk value at that locus must not affect
`L`, and the imputed placeholder must not be `0.0` (criterion 7).

## Test 10: unreachable DMR is dropped

A DMR whose every locus is missing in every reference is dropped (`is_dmr = 0`,
delta zeroed inside it, `dmr_id` reported); a DMR with at least one reachable
locus is kept.

## Test 11: reconstruction error switch

With full coverage weight, `recon_loss = mse` equals the analytic weighted MSE
and masking an entry reduces it to that entry alone; `huber` stays the default.

---

# 22. Example config

```yaml
seed: 123

data:
  methylation_scale: 1.0
  use_coverage: true
  min_coverage: 5
  dmr_only_delta: true

reference:
  beta_prior_a0: 0.5
  beta_prior_b0: 0.5
  concentration_if_no_coverage: 50.0

composition:
  mode: dirichlet_prior
  prior_strength: 100.0
  epsilon: 1.0e-4
  aggregate_unreferenced_as_other: true

other:
  enabled: true
  min_fraction_for_residual_init: 0.02
  anchor_weight: 1.0
  smooth_weight: 0.01

model:
  version: deterministic_group_delta
  delta_scope: dmr_only
  delta_l1_weight: 0.01
  delta_group_weight: 0.0

likelihood:
  type: huber
  huber_delta: 0.05
  coverage_cap: 30

reconstruction:
  control_weight: 10.0
  disease_weight: 10.0
  target_mae: 0.03
  target_rmse: 0.05

training:
  lr: 1.0e-3
  weight_decay: 1.0e-5
  max_epochs_stage2: 2000
  max_epochs_stage3: 3000
  early_stopping_patience: 100
  checkpoint_every: 50

split:
  strategy: chromosome
```

---

# 23. Run script behavior

`run_all.sh` must:

1. validate inputs
2. build aligned matrices
3. fit reference distributions
4. fit control model
5. stop if control QC fails
6. fit disease model
7. calculate attribution
8. run QC
9. save final outputs

Required console logging:

```text
[STEP 0] Input validation
[STEP 1] Reference fitting
[STEP 2] Control mixture fitting
[STEP 2] CTRL RMSE=...
[STEP 3] Disease delta fitting
[STEP 3] DIS RMSE=...
[STEP 4] Attribution
[STEP 5] QC
[DONE]
```

Each stage must create a checkpoint and completion marker.

`resume.sh` must continue from the latest completed stage.

---

# 24. Acceptance criteria

The first implementation is successful only if:

1. All predicted methylation values are within `[0,1]`.
2. All proportions are non-negative and sum to 1.
3. CTRL prediction is constructed exclusively from baseline cell-type tracks and proportions.
4. DIS prediction is constructed exclusively from perturbed cell-type tracks and proportions.
5. Disease delta is added before mixture.
6. Missing reference fractions are explicitly assigned to OTHER.
7. No missing methylation value is silently treated as true zero.
8. Synthetic mixtures are reconstructed correctly.
9. Synthetic injected disease deltas are recovered with correct sign and responsible cell type at above-random accuracy.
10. Control reconstruction passes tolerance before disease attribution is interpreted.
11. Sensitivity to composition prior and OTHER is reported.
12. Results are reproducible under a fixed seed.
13. A masked `(cell_type, locus)` entry cannot influence the objective: changing its
    value, or the delta fitted on it, leaves `L` unchanged.
14. A locus that no reference observes is excluded from `L_recon`, and a DMR whose
    every locus is in that state is dropped as a unit and reported.

---

# 25. Interpretation limits

The model estimates a statistically constrained cell-type attribution of bulk methylation changes.

Use:

- `inferred cell-type-specific methylation state`
- `reference-anchored disease perturbation`
- `cell-type contribution to bulk DMR`
- `composition-associated component`
- `cell-intrinsic methylation component`

Avoid claiming:

- reconstructed tracks are directly observed truth
- delta is a biochemical reaction rate
- attribution is causal without orthogonal validation

---

# 26. Optional future integration with Cross-Hyena

After the mixture model is validated, replace free locus-level delta with a structured decoder.

For cell type \(c\), locus \(r\):

\[
h_{c,r}
=
CrossHyena(
Sequence_r,
ATAC_{c,r},
MethylationReference_{c,r}
)
\]

Then:

\[
\Delta_{s,c,r}
=
g_\theta(
z_s,
e_c,
h_{c,r}
)
\]

This future extension allows sequence context, cell-type ATAC, control methylation reference and latent disease state to jointly predict cell-type-specific disease perturbation.

---

# 27. Minimal implementation order

Implement in this exact order:

```text
A. deterministic mixture with fixed pi, known references only
B. add OTHER component
C. add learned pi with single-cell Dirichlet prior
D. add sparse disease delta on DMRs
E. add decomposition and attribution
F. add synthetic recovery benchmark
G. add reference uncertainty / Beta sampling
H. add optional Beta-Binomial likelihood
I. add VAE latent z
J. optionally integrate Cross-Hyena / ATAC / sequence
```

Do not implement I or J before A-F pass unit tests and synthetic benchmarks.

---

# 28. One-sentence model definition

> A reference-anchored probabilistic mixture model in which control bulk methylation is reconstructed as a simplex-weighted mixture of cell-type control methylomes, unreferenced cell populations are represented by an explicitly regularized OTHER component informed by single-cell composition priors, and disease bulk methylation is reconstructed after cell-type-specific methylation perturbations are applied in logit space before mixing.

"""Reference-anchored cell-type methylation mixture model (deterministic v1).

Implements stages A-E of `reference_anchored_methylation_mixture_model_spec.md`
§27:

    A. deterministic mixture with known references
    B. explicit OTHER component for unreferenced cell types
    C. learnable composition with a single-cell Dirichlet prior
    D. sparse disease delta, applied in logit space BEFORE mixing, on DMRs
    E. composition / intrinsic-effect attribution

Deliberately NOT implemented here (spec §10, §26): the VAE latent disease state
and the Cross-Hyena / ATAC / sequence decoder.  Those come after A-E pass the
unit tests and the synthetic benchmark (spec §27).

The two architectural identities from the spec are enforced by construction:

    y_hat_CTRL[s, r] = sum_c pi_CTRL[s, c] * theta0[c, r]
    y_hat_DIS [s, r] = sum_c pi_DIS [s, c] * sigmoid(logit(theta0[c, r]) + delta[c, r])

with ``delta`` restricted to DMR loci (``delta_scope='dmr_only'``) and never added
after the mixture.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

OTHER = "OTHER"


def _logit(p: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.log(p.clamp(eps, 1.0 - eps) / (1.0 - p.clamp(eps, 1.0 - eps)))


def _sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable logistic (uses the logit-space inputs we feed it)."""
    return torch.sigmoid(x)


@dataclass
class MixtureConfig:
    """Hyperparameters mirroring the spec's example config (§22)."""

    eps: float = 1.0e-4                      # clamp for logit(theta) and pi floors
    composition_mode: str = "dirichlet_prior"  # "dirichlet_prior" | "fixed"
    composition_prior_strength: float = 100.0  # kappa_pi
    composition_epsilon: float = 1.0e-4
    use_other: bool = True                     # OTHER sensitivity runs set this False
    delta_scope: str = "dmr_only"              # "dmr_only" | "all_loci"
    # loss weights (spec §12 / §22)
    lambda_ctrl_recon: float = 10.0
    lambda_dis_recon: float = 10.0
    lambda_composition: float = 1.0
    lambda_delta_l1: float = 0.01
    lambda_delta_group: float = 0.0
    # Common-mode penalty: |sum_c delta[c, r]| . A perturbation shared by every
    # cell type is indistinguishable from a single-cell-type perturbation in the
    # bulk (they differ only by a factor sum_c pi_c = 1), so L1 alone cannot rank
    # the responsible cell type. Penalising the common mode makes concentrated
    # explanations cheaper (spec §24 criterion 9). Default 0 keeps the loss list
    # of spec §12 unchanged; enable it for attribution/rank-recovery runs.
    lambda_delta_common: float = 0.0
    lambda_other_anchor: float = 1.0
    lambda_smooth: float = 0.0
    lambda_delta_other_l1: float = 0.1        # extra sparsity on delta[OTHER] (spec §18)
    huber_delta: float = 0.05
    coverage_cap: float = 30.0
    # stage-2/3 quality targets (spec §22)
    target_mae: float = 0.03
    target_rmse: float = 0.05
    backbone_eps: float = 1.0e-6


@dataclass
class MixtureBatch:
    """Aligned tensors for one analysis group (all shapes documented inline).

    y_*       : [S, R] observed methylation in [0, 1]
    mask_*    : [S, R] bool, True where the observation exists (spec §3.1: missing
                is encoded by mask=0, never by value 0)
    weight_*  : [S, R] coverage weight (>=0); pass all-ones when unavailable
    pi0_*     : [S, C] composition prior (already includes the OTHER column)
    """

    y_ctrl: torch.Tensor | None = None
    mask_ctrl: torch.Tensor | None = None
    weight_ctrl: torch.Tensor | None = None
    pi0_ctrl: torch.Tensor | None = None
    y_dis: torch.Tensor | None = None
    mask_dis: torch.Tensor | None = None
    weight_dis: torch.Tensor | None = None
    pi0_dis: torch.Tensor | None = None

    def to(self, device) -> "MixtureBatch":
        out = {}
        for key, value in self.__dict__.items():
            out[key] = value.to(device) if value is not None else None
        return MixtureBatch(**out)


class ReferenceAnchoredMethylationMixture(nn.Module):
    """Deterministic mixture model for bulk methylation deconvolution.

    Parameters
    ----------
    theta_ref : [C_ref, R]
        Known reference cell-type methylation means in (0, 1) (spec §4). Frozen.
    theta_ref_conc : [C_ref, R] or scalar, optional
        Beta concentration of each reference (α+β). Kept for the optional
        Beta-Binomial likelihood (spec §27 H); unused by the deterministic loss.
    pi0_ctrl / pi0_dis : [S_ctrl, C] / [S_dis, C]
        Single-cell composition priors over components, where the last column is
        OTHER = 1 - sum(referenced) (spec §6).
    other_init : [R], optional
        Control-residual initialisation of the OTHER baseline (spec §6). When
        omitted, OTHER is initialised to the weighted mean of the references.
    is_dmr : [R] bool, optional
        DMR annotation; with ``delta_scope='dmr_only'`` the disease delta is
        masked to these loci.
    """

    def __init__(
        self,
        theta_ref: torch.Tensor,
        pi0_ctrl: torch.Tensor,
        pi0_dis: torch.Tensor | None = None,
        other_init: torch.Tensor | None = None,
        is_dmr: torch.Tensor | None = None,
        theta_ref_conc: torch.Tensor | float | None = None,
        cfg: MixtureConfig | None = None,
        cell_type_names: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or MixtureConfig()
        eps = self.cfg.eps

        if theta_ref.dim() != 2:
            raise ValueError(f"theta_ref must be [C_ref, R], got {tuple(theta_ref.shape)}")
        theta_ref = theta_ref.float().clamp(eps, 1.0 - eps)
        n_ref, n_loci = theta_ref.shape
        self.n_ref = n_ref
        self.n_loci = n_loci
        self.n_components = n_ref + (1 if self.cfg.use_other else 0)

        self.register_buffer("theta_ref", theta_ref)
        if isinstance(theta_ref_conc, torch.Tensor):
            self.register_buffer("theta_ref_conc", theta_ref_conc.float())
        else:
            self.register_buffer("theta_ref_conc", torch.full_like(theta_ref, float(theta_ref_conc or 0.0)))
        if is_dmr is None:
            is_dmr = torch.ones(n_loci, dtype=torch.bool)
        self.register_buffer("is_dmr", is_dmr.bool())

        # ---- composition -------------------------------------------------
        if pi0_ctrl.shape[1] != self.n_components:
            raise ValueError(
                f"pi0_ctrl must have {self.n_components} columns (references"
                f"{' + OTHER' if self.cfg.use_other else ''}), got {pi0_ctrl.shape[1]}"
            )
        pi0_ctrl = pi0_ctrl.float().clamp_min(self.cfg.composition_epsilon)
        pi0_ctrl = pi0_ctrl / pi0_ctrl.sum(dim=-1, keepdim=True)
        if pi0_dis is None:
            pi0_dis = pi0_ctrl.clone()
        pi0_dis = pi0_dis.float().clamp_min(self.cfg.composition_epsilon)
        pi0_dis = pi0_dis / pi0_dis.sum(dim=-1, keepdim=True)

        self.register_buffer("pi0_ctrl", pi0_ctrl)
        self.register_buffer("pi0_dis", pi0_dis)
        if self.cfg.composition_mode == "dirichlet_prior":
            # §20: pi = softmax(pi_logits); start at the single-cell prior.
            self.pi_ctrl_logits = nn.Parameter(pi0_ctrl.clamp_min(eps).log())
            self.pi_dis_logits = nn.Parameter(pi0_dis.clamp_min(eps).log())
        elif self.cfg.composition_mode == "fixed":
            self.register_buffer("pi_ctrl_fixed", pi0_ctrl)
            self.register_buffer("pi_dis_fixed", pi0_dis)
        else:
            raise ValueError(f"unknown composition_mode: {self.cfg.composition_mode}")

        # ---- OTHER baseline ----------------------------------------------
        if self.cfg.use_other:
            if other_init is None:
                # Fall back to the reference mean-weighted by the control prior.
                weights = pi0_ctrl.mean(dim=0)[:n_ref]
                weights = weights / weights.sum().clamp_min(eps)
                other_init = torch.einsum("c,cr->r", weights, theta_ref)
            other_logits_init = _logit(other_init.float().clamp(eps, 1.0 - eps), eps)
            self.other_logits = nn.Parameter(other_logits_init.clone())
            self.register_buffer("other_logits_init", other_logits_init.clone())

        # ---- disease delta ------------------------------------------------
        # zero-initialised => theta_dis == theta0 at start (spec §20, test 4)
        self.delta = nn.Parameter(torch.zeros(self.n_components, n_loci))

        # ---- names ---------------------------------------------------------
        if cell_type_names is None:
            cell_type_names = [f"celltype_{i}" for i in range(n_ref)]
        self.cell_type_names = list(cell_type_names)
        if self.cfg.use_other:
            self.cell_type_names.append(OTHER)

    # ------------------------------------------------------------------ #
    # component tracks
    # ------------------------------------------------------------------ #
    def theta0(self) -> torch.Tensor:
        """[C, R] control-state cell-type methylomes (references + OTHER)."""
        if not self.cfg.use_other:
            return self.theta_ref
        return torch.cat([self.theta_ref, torch.sigmoid(self.other_logits)[None, :]], dim=0)

    def delta_effective(self) -> torch.Tensor:
        """[C, R] disease logit shift, restricted to DMR loci when configured."""
        if self.cfg.delta_scope == "dmr_only":
            return self.delta * self.is_dmr.to(self.delta.dtype)[None, :]
        if self.cfg.delta_scope == "all_loci":
            return self.delta
        raise ValueError(f"unknown delta_scope: {self.cfg.delta_scope}")

    def theta_disease(self) -> torch.Tensor:
        """[C, R] perturbed methylomes: sigmoid(logit(theta0) + delta) (spec §8)."""
        eps = self.cfg.eps
        return torch.sigmoid(_logit(self.theta0(), eps) + self.delta_effective())

    def composition(self, condition: str) -> torch.Tensor:
        """[S, C] simplex-valued cell-type fractions for `condition`."""
        if self.cfg.composition_mode == "fixed":
            return self.pi_ctrl_fixed if condition == "CTRL" else self.pi_dis_fixed
        logits = self.pi_ctrl_logits if condition == "CTRL" else self.pi_dis_logits
        return torch.softmax(logits, dim=-1)

    # ------------------------------------------------------------------ #
    # the two required architectural identities (spec §7, §8)
    # ------------------------------------------------------------------ #
    def predict_control(self) -> torch.Tensor:
        """[S_ctrl, R] = sum_c pi_CTRL[s, c] * theta0[c, r]."""
        return torch.einsum("sc,cr->sr", self.composition("CTRL"), self.theta0())

    def predict_disease(self) -> torch.Tensor:
        """[S_dis, R] = sum_c pi_DIS[s, c] * theta_disease[c, r]."""
        return torch.einsum("sc,cr->sr", self.composition("DIS"), self.theta_disease())

    # ------------------------------------------------------------------ #
    # losses (spec §11, §12, §13)
    # ------------------------------------------------------------------ #
    def _recon(
        self,
        pred: torch.Tensor,
        observed: torch.Tensor,
        mask: torch.Tensor,
        weight: torch.Tensor | None,
    ) -> torch.Tensor:
        """Masked, coverage-weighted Huber loss over valid observations."""
        cfg = self.cfg
        err = F.huber_loss(pred, observed, reduction="none", delta=cfg.huber_delta)
        m = mask.float()
        if weight is not None:
            m = m * (weight.clamp(max=cfg.coverage_cap) / cfg.coverage_cap).float()
        denom = m.sum().clamp_min(1.0)
        return (err * m).sum() / denom

    def _composition_nll(self, condition: str) -> torch.Tensor:
        """-log p(pi | Dirichlet(kappa * prior)) (spec §12)."""
        cfg = self.cfg
        prior = self.pi0_ctrl if condition == "CTRL" else self.pi0_dis
        concentration = (cfg.composition_prior_strength * prior + cfg.composition_epsilon).clamp_min(
            cfg.backbone_eps
        )
        pi = self.composition(condition).clamp_min(cfg.backbone_eps)
        return -(torch.distributions.Dirichlet(concentration).log_prob(pi)).mean()

    def _other_anchor(self) -> torch.Tensor:
        """Keep OTHER near its control-residual init (spec §6)."""
        if not self.cfg.use_other:
            return self.theta_ref.new_zeros(())
        eps = self.cfg.eps
        return F.mse_loss(_logit(torch.sigmoid(self.other_logits), eps),
                          self.other_logits_init, reduction="mean")

    def _delta_sparsity(self) -> tuple[torch.Tensor, torch.Tensor]:
        """L1 on delta (extra weight on the OTHER row) and optional group L2."""
        cfg = self.cfg
        delta_eff = self.delta_effective()
        if self.cfg.use_other:
            other_row = delta_eff[-1].abs().mean() * cfg.lambda_delta_other_l1 * self.n_loci
            ref_l1 = delta_eff[:-1].abs().mean() * max(self.n_ref, 1)
            l1 = (ref_l1 + other_row) / max(self.n_components, 1)
        else:
            l1 = delta_eff.abs().mean()
        group = torch.linalg.vector_norm(delta_eff, dim=0).mean() if delta_eff.numel() else delta_eff.new_zeros(())
        return l1, group

    def _delta_common_mode(self) -> torch.Tensor:
        """|sum_c delta_eff[c, r]| averaged over loci (identifiability penalty).

        Absent this term, "every cell type shifts a little" and "one cell type
        shifts a lot" fit the bulk equally well; the penalty prefers the sparse
        (concentrated) explanation, which is what cell-type attribution needs.
        """
        if self.cfg.lambda_delta_common <= 0:
            return self.theta_ref.new_zeros(())
        return self.delta_effective().sum(dim=0).abs().mean()

    def _smooth(self) -> torch.Tensor:
        if self.cfg.lambda_smooth <= 0 or self.n_loci < 2:
            return self.theta_ref.new_zeros(())
        d = self.delta_effective()
        return (d[:, 1:] - d[:, :-1]).abs().mean()

    def loss(self, batch: MixtureBatch, stage: str = "joint") -> tuple[torch.Tensor, dict]:
        """Total loss for `batch`. `stage` selects which terms are active.

        stage='ctrl' -> control reconstruction + composition + OTHER anchor
        stage='dis'  -> control recon (guard) + disease recon + delta penalties
        stage='joint'-> everything
        """
        cfg = self.cfg
        terms: dict[str, torch.Tensor] = {}
        total = self.theta_ref.new_zeros(())

        if batch.y_ctrl is not None and stage in ("ctrl", "dis", "joint"):
            lc = self._recon(self.predict_control(), batch.y_ctrl, batch.mask_ctrl, batch.weight_ctrl)
            terms["recon_ctrl"] = lc
            total = total + cfg.lambda_ctrl_recon * lc
            lpi_c = self._composition_nll("CTRL")
            terms["composition_ctrl"] = lpi_c
            total = total + cfg.lambda_composition * lpi_c
            anchor = self._other_anchor()
            terms["other_anchor"] = anchor
            total = total + cfg.lambda_other_anchor * anchor

        if batch.y_dis is not None and stage in ("dis", "joint"):
            ld = self._recon(self.predict_disease(), batch.y_dis, batch.mask_dis, batch.weight_dis)
            terms["recon_dis"] = ld
            total = total + cfg.lambda_dis_recon * ld
            lpi_d = self._composition_nll("DIS")
            terms["composition_dis"] = lpi_d
            total = total + cfg.lambda_composition * lpi_d

        if stage in ("dis", "joint"):
            l1, group = self._delta_sparsity()
            terms["delta_l1"] = l1
            terms["delta_group"] = group
            total = total + cfg.lambda_delta_l1 * l1 + cfg.lambda_delta_group * group
            common = self._delta_common_mode()
            terms["delta_common"] = common
            total = total + cfg.lambda_delta_common * common
            smooth = self._smooth()
            terms["delta_smooth"] = smooth
            total = total + cfg.lambda_smooth * smooth

        terms["total"] = total
        return total, terms

    # ------------------------------------------------------------------ #
    # attribution (spec §14)
    # ------------------------------------------------------------------ #
    def attribution(self) -> dict[str, torch.Tensor]:
        """Decompose the bulk disease effect into composition + intrinsic parts."""
        with torch.no_grad():
            theta0 = self.theta0()
            theta_dis = self.theta_disease()
            pi_c = self.composition("CTRL").mean(dim=0)   # [C]
            pi_d = self.composition("DIS").mean(dim=0)    # [C]
            delta_pi = pi_d - pi_c
            composition_effect = torch.einsum("c,cr->r", delta_pi, theta0)
            intrinsic_by_cell = pi_d[:, None] * (theta_dis - theta0)   # [C, R]
            intrinsic_effect = intrinsic_by_cell.sum(dim=0)
            return {
                "pi_ctrl_mean": pi_c,
                "pi_dis_mean": pi_d,
                "theta0": theta0,
                "theta_dis": theta_dis,
                "delta_effective": self.delta_effective(),
                "composition_effect": composition_effect,
                "intrinsic_effect": intrinsic_effect,
                "celltype_intrinsic": intrinsic_by_cell,
                "bulk_delta": composition_effect + intrinsic_effect,
            }


# ---------------------------------------------------------------------- #
# stage fitting (spec §18)
# ---------------------------------------------------------------------- #
@dataclass
class FitHistory:
    losses: list[float] = field(default_factory=list)
    metrics: list[dict] = field(default_factory=list)


def _masked_metrics(
    pred: torch.Tensor,
    observed: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor | None,
    tolerance: float,
) -> dict:
    """RMSE / MAE / Pearson / within-tolerance fraction over valid entries."""
    m = mask.bool()
    if m.sum() == 0:
        return {"n": 0, "rmse": float("nan"), "mae": float("nan"), "pearson": float("nan"),
                "within_tolerance": float("nan")}
    p = pred[m]
    y = observed[m]
    w = None
    if weight is not None:
        w = weight[m].clamp_min(0.0)
    err = (p - y).abs()
    if w is not None and w.sum() > 0:
        rmse = float(torch.sqrt((((p - y) ** 2) * w).sum() / w.sum()))
        mae = float((err * w).sum() / w.sum())
        within = float(((err <= tolerance).float() * w).sum() / w.sum())
    else:
        rmse = float(torch.sqrt(((p - y) ** 2).mean()))
        mae = float(err.mean())
        within = float((err <= tolerance).float().mean())
    pc = p - p.mean()
    yc = y - y.mean()
    denom = float(torch.sqrt((pc ** 2).sum() * (yc ** 2).sum()))
    pearson = float((pc * yc).sum() / denom) if denom > 0 else float("nan")
    return {"n": int(m.sum()), "rmse": rmse, "mae": mae, "pearson": pearson, "within_tolerance": within}


def fit_stage(
    model: ReferenceAnchoredMethylationMixture,
    batch: MixtureBatch,
    stage: str,
    lr: float = 1.0e-3,
    weight_decay: float = 1.0e-5,
    max_epochs: int = 2000,
    patience: int = 100,
    log_every: int = 0,
    verbose: bool = True,
) -> FitHistory:
    """Optimise the parameters relevant to `stage` (spec §18 stages 2 and 3).

    stage='ctrl': trains pi_CTRL (if learnable) and the OTHER baseline.
    stage='dis' : trains pi_DIS and the DMR delta (theta0 / OTHER stay fixed).
    """
    freeze_all(model)
    if stage == "ctrl":
        trainable = []
        if model.cfg.composition_mode == "dirichlet_prior":
            model.pi_ctrl_logits.requires_grad_(True)
            trainable.append(model.pi_ctrl_logits)
        if model.cfg.use_other:
            model.other_logits.requires_grad_(True)
            trainable.append(model.other_logits)
    elif stage == "dis":
        model.delta.requires_grad_(True)
        trainable = [model.delta]
        if model.cfg.composition_mode == "dirichlet_prior":
            model.pi_dis_logits.requires_grad_(True)
            trainable.append(model.pi_dis_logits)
    elif stage == "both":
        trainable = [p for p in model.parameters()]
        for p in trainable:
            p.requires_grad_(True)
    else:
        raise ValueError(f"unknown stage: {stage}")
    if not trainable:
        raise RuntimeError(f"stage '{stage}' has no trainable parameters")

    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    history = FitHistory()
    best_loss = float("inf")
    best_state = None
    patience_left = patience

    for epoch in range(1, max_epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, terms = model.loss(batch, stage=stage)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at epoch {epoch}: {terms}")
        loss.backward()
        optimizer.step()

        value = float(loss.detach())
        history.losses.append(value)
        if value < best_loss - 1e-9:
            best_loss = value
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
        if log_every and epoch % log_every == 0:
            history.metrics.append({"epoch": epoch, **{k: float(v) for k, v in terms.items()}})
            if verbose:
                recon = terms.get("recon_ctrl", terms.get("recon_dis"))
                print(f"  [{stage}] epoch {epoch:5d} loss={value:.6f} recon={float(recon):.6f}")
        if patience_left <= 0:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def freeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad_(False)


def stage_metrics(
    model: ReferenceAnchoredMethylationMixture, batch: MixtureBatch, tolerance: float | None = None
) -> dict:
    """Reconstruction QC for the current parameters (spec §15 model_qc)."""
    tol = model.cfg.target_mae if tolerance is None else tolerance
    out: dict[str, dict] = {}
    with torch.no_grad():
        if batch.y_ctrl is not None:
            out["control"] = _masked_metrics(
                model.predict_control(), batch.y_ctrl, batch.mask_ctrl, batch.weight_ctrl, tol
            )
        if batch.y_dis is not None:
            out["disease"] = _masked_metrics(
                model.predict_disease(), batch.y_dis, batch.mask_dis, batch.weight_dis, tol
            )
        delta = model.delta_effective()
        out["delta_sparsity"] = {
            "fraction_nonzero": float((delta.abs() > 1e-8).float().mean()),
            "l1_mean": float(delta.abs().mean()),
            "l2_group_mean": float(torch.linalg.vector_norm(delta, dim=0).mean()),
        }
        pi_d = model.composition("DIS")
        prior_d = model.pi0_dis
        out["composition_prior_deviation"] = {
            "mean_abs_deviation": float((pi_d - prior_d).abs().mean()),
            "other_fraction_mean": float(pi_d[:, -1].mean()) if model.cfg.use_other else 0.0,
        }
    return out

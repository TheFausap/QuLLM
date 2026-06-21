"""Coherence-isolation variants for the TinyStories attention probe.

WHY
---
The existing decohere/frozen/free_mix controls all collapse to ~0.67 because in
ComplexAttentionStackedProbe the phase path carries TWO things that have nothing
to do with coherence:

  1. POSITION.  Position enters only via pos_phase rotation (init zeros), gated on
     `not force_zero_phase`.  Turn phase off -> the model is position-blind.
  2. TRANSFORM CAPACITY.  The only q/k/v/out transforms ARE the diagonal phase
     rotations; force_zero_phase removes them entirely, leaving parameter-free
     attention, versus RealAttentionLayer's full dim x dim projections.

So intact-minus-decohere measures "phase carries position + transforms", not
"coherence buys prediction".  To isolate coherence we need a twin that differs
from the intact model in ONE thing only: the real<->imag coupling (interference).

WHAT THIS ADDS
--------------
A position-aware coherent probe and a MATCHED decohere twin:

  complex_attention_stacked_floor_pos            (coherent)
  complex_attention_stacked_floor_pos_decohere   (decohered, matched)

Both share, held EQUAL:
  * additive real positional channel (phase-independent; pos_phase is NOT used)
  * identical per-dim transform parameter budget (q/k/v/out + readout)
  * identical mixing, readout, init, depth behaviour, param count

The ONLY difference:
  * coherent  : each transform is a per-dim 2D ROTATION  [[c,-s],[s,c]]  -> couples
                real & imag  -> interference ON.
  * decohere  : each transform is a per-dim REAL DIAGONAL gain [[a,0],[0,a]] -> no
                real<->imag coupling -> interference OFF, same parameter count.

If, with this twin, intact-minus-decohere collapses to ~0.02 and the decohere
lands near real_attention_stacked, the control is clean and coherence's
contribution is isolated and bounded.  If the decohere instead matches the
intact model, coherence is doing nothing here and the earlier complex-vs-real
edge was the diagonal-transform / weight-sharing efficiency, not phase.

INTEGRATION
-----------
This module registers the two new model names into the probe's MODEL_TYPES at
import.  In tinystories_attention_probe.py add ONE line near the top, after
MODEL_TYPES is defined (or in your runner before models are built):

    import experiments.coherence_isolation  # noqa: F401  (registers _pos variants)

then run e.g.

    --models complex_attention_stacked_floor_pos,complex_attention_stacked_floor_pos_decohere,real_attention_stacked

Run this file directly for a self-contained mechanics check on a position-
sensitive synthetic task (no TinyStories needed):

    python3 experiments/coherence_isolation.py
"""
from __future__ import annotations

import math

import torch
from torch import nn


class CoherenceLayer(nn.Module):
    """One stacked-attention layer with a switchable transform mode.

    mode='rotate'  : per-dim 2D rotation (interference ON)   -- the coherent model
    mode='diag'    : per-dim real diagonal gain (interference OFF) -- matched twin

    Both modes use exactly `dim` parameters per transform (q/k/v/out), so the two
    models are parameter-matched.
    """

    def __init__(self, dim: int, mode: str = "rotate", phase_init: float = 0.05,
                 min_mix: float = 0.2, mix_init: float = -2.0) -> None:
        super().__init__()
        assert mode in ("rotate", "diag")
        self.mode = mode
        # one dim-vector per transform: angles (rotate) or log-gains (diag)
        self.q_t = nn.Parameter(torch.zeros(dim))
        self.k_t = nn.Parameter(torch.zeros(dim))
        self.v_t = nn.Parameter(torch.zeros(dim))
        self.out_t = nn.Parameter(torch.zeros(dim))
        if phase_init > 0:
            for p in (self.q_t, self.k_t, self.v_t, self.out_t):
                nn.init.normal_(p, std=phase_init)
        self.mix = nn.Parameter(torch.tensor(mix_init))
        self.min_mix = min_mix

    def transform(self, real, imag, t):
        if self.mode == "rotate":
            c, s = torch.cos(t), torch.sin(t)
            return real * c - imag * s, real * s + imag * c
        # diag: real per-dim gain applied to each channel independently (no coupling)
        g = torch.exp(t)  # init ~1.0; strictly positive, learnable
        return real * g, imag * g

    @staticmethod
    def normalize(real, imag):
        scale = torch.sqrt((real.square() + imag.square()).sum(-1, keepdim=True).clamp_min(1e-8))
        return real / scale, imag / scale

    def forward(self, c_real, c_imag, x_real, x_imag):
        q_real, q_imag = self.transform(c_real, c_imag, self.q_t)
        k_real, k_imag = self.transform(x_real, x_imag, self.k_t)
        v_real, v_imag = self.transform(x_real, x_imag, self.v_t)
        q_real, q_imag = self.normalize(q_real, q_imag)
        k_real, k_imag = self.normalize(k_real, k_imag)
        compat = (q_real.unsqueeze(1) * k_real + q_imag.unsqueeze(1) * k_imag).sum(-1)
        attn = torch.softmax(compat * math.sqrt(k_real.shape[-1]), dim=-1)
        pooled_real = (attn.unsqueeze(-1) * v_real).sum(1)
        pooled_imag = (attn.unsqueeze(-1) * v_imag).sum(1)
        pooled_real, pooled_imag = self.transform(pooled_real, pooled_imag, self.out_t)
        mix = self.min_mix + (1.0 - self.min_mix) * torch.sigmoid(self.mix)
        return self.normalize(c_real + mix * pooled_real, c_imag + mix * pooled_imag)


class CoherenceStackedProbe(nn.Module):
    """Position-aware stacked complex attention with switchable interference.

    Position is an ADDITIVE real embedding on the context real channel (mirrors
    RealAttentionStackedProbe), phase-independent, identical for both modes.
    """

    def __init__(self, vocab_size: int, dim: int, context: int, layers: int = 2,
                 mode: str = "rotate", phase_init: float = 0.05, min_mix: float = 0.2) -> None:
        super().__init__()
        self.mode = mode
        self.real = nn.Embedding(vocab_size, dim)
        self.imag = nn.Embedding(vocab_size, dim)
        self.pos = nn.Embedding(context, dim)            # additive, phase-independent
        self.readout_t = nn.Parameter(torch.zeros(dim))  # rotate-angle or log-gain
        self.layers = nn.ModuleList(
            CoherenceLayer(dim, mode=mode, phase_init=phase_init, min_mix=min_mix)
            for _ in range(layers)
        )
        self.real_weight = nn.Parameter(torch.tensor(1.0))
        self.imag_weight = nn.Parameter(torch.tensor(0.0))
        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        self.logit_bias = nn.Parameter(torch.tensor(0.0))
        nn.init.normal_(self.real.weight, std=(2 * dim) ** -0.5)
        nn.init.normal_(self.imag.weight, std=(2 * dim) ** -0.5)
        nn.init.normal_(self.pos.weight, std=(2 * dim) ** -0.5)
        # phase_active/phase_trainable telemetry hooks for the existing summarizer
        self.force_zero_phase = False
        self.phase_trainable = (mode == "rotate")

    def _readout(self, real, imag):
        if self.mode == "rotate":
            c, s = torch.cos(self.readout_t), torch.sin(self.readout_t)
            return real * c - imag * s, real * s + imag * c
        g = torch.exp(self.readout_t)
        return real * g, imag * g

    @staticmethod
    def normalize(real, imag):
        scale = torch.sqrt((real.square() + imag.square()).sum(-1, keepdim=True).clamp_min(1e-8))
        return real / scale, imag / scale

    def forward(self, context_ids, candidate):
        positions = torch.arange(context_ids.shape[1], device=context_ids.device)
        x_real = self.real(context_ids) + self.pos(positions)   # additive position
        x_imag = self.imag(context_ids)
        base_real = self.real(candidate)
        base_imag = self.imag(candidate)
        c_real, c_imag = self.normalize(base_real, base_imag)
        for layer in self.layers:
            c_real, c_imag = layer(c_real, c_imag, x_real, x_imag)
        c_real, c_imag = self._readout(c_real, c_imag)
        inner_real = (c_real * base_real + c_imag * base_imag).sum(-1)
        inner_imag = (c_real * base_imag - c_imag * base_real).sum(-1)
        score = (self.real_weight * inner_real + self.imag_weight * inner_imag) / math.sqrt(c_real.shape[-1])
        return self.logit_scale * score + self.logit_bias


class CoherentFloorPosProbe(CoherenceStackedProbe):
    def __init__(self, vocab_size, dim, context, layers=2):
        super().__init__(vocab_size, dim, context, layers, mode="rotate")


class DecohereFloorPosProbe(CoherenceStackedProbe):
    def __init__(self, vocab_size, dim, context, layers=2):
        super().__init__(vocab_size, dim, context, layers, mode="diag")


def register():
    """Register the two new variants into the probe's MODEL_TYPES."""
    from experiments import tinystories_attention_probe as P
    P.MODEL_TYPES["complex_attention_stacked_floor_pos"] = CoherentFloorPosProbe
    P.MODEL_TYPES["complex_attention_stacked_floor_pos_decohere"] = DecohereFloorPosProbe


try:  # auto-register on import when the probe module is importable
    register()
except BaseException:
    pass


# --------------------------------------------------------------------------- #
# Self-contained mechanics check (no TinyStories): a POSITION-SENSITIVE task.
# label = 1 iff candidate == context[target_pos].  A position-blind model can
# only tell "candidate is somewhere in context"; a position-aware model can
# pin the target slot.  This validates that (a) the additive-pos twin recovers
# position, (b) intact vs decohere is parameter-matched, (c) both train.
# --------------------------------------------------------------------------- #
def _no_pos(model):
    """Position-blind control: zero and freeze the additive positional channel."""
    with torch.no_grad():
        model.pos.weight.zero_()
    model.pos.weight.requires_grad_(False)
    return model


def _make_data(n, vocab, context, target_pos, device, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    ctx = torch.randint(0, vocab, (n, context), generator=g)
    pos_label = torch.rand(n, generator=g) < 0.5
    cand = torch.randint(0, vocab, (n,), generator=g)
    cand = torch.where(pos_label, ctx[:, target_pos], cand)
    y = (cand == ctx[:, target_pos]).float()
    return ctx.to(device), cand.to(device), y.to(device)


def _train_eval(model, tr, te, device, epochs=12, bs=512, lr=5e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    cx, cd, y = tr
    n = y.numel()
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for s in range(0, n, bs):
            i = perm[s:s + bs]
            loss = nn.functional.binary_cross_entropy_with_logits(model(cx[i], cd[i]), y[i])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    with torch.no_grad():
        cx2, cd2, y2 = te
        acc = ((model(cx2, cd2) > 0).float() == y2).float().mean().item()
    return acc, sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    torch.manual_seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    V, D, C, L, TP = 256, 32, 8, 6, 3
    tr = _make_data(20000, V, C, TP, dev, 1)
    te = _make_data(8000, V, C, TP, dev, 2)
    print(f"position-sensitive check  vocab={V} dim={D} context={C} layers={L} target_pos={TP}  device={dev}")
    builders = {
        "coherent_pos (rotate)":      lambda: CoherenceStackedProbe(V, D, C, L, mode="rotate"),
        "decohere_pos (diag)":        lambda: CoherenceStackedProbe(V, D, C, L, mode="diag"),
        "decohere_NO_pos (control)":  lambda: _no_pos(CoherenceStackedProbe(V, D, C, L, mode="diag")),
    }
    for name, build in builders.items():
        accs = []
        for seed in range(3):
            torch.manual_seed(seed)
            m = build().to(dev)
            acc, npar = _train_eval(m, tr, te, dev)
            accs.append(acc)
        mu = sum(accs) / len(accs)
        sd = (sum((a - mu) ** 2 for a in accs) / len(accs)) ** 0.5
        print(f"  {name:28s} acc={mu:.3f}+/-{sd:.3f}  params={npar}")

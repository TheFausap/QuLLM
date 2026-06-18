# Results Notes

## Structured Synthetic Phase Probe

The structured synthetic task confirmed that relation phase can be strongly scale dependent.

At the largest DGX setting, `phase_margin_fixed` reached perfect accuracy with only 16 trainable relation-phase parameters:

```text
phase_margin_fixed accuracy=1.0000 params=16
real_feature_mlp   accuracy=1.0000 params=75457
```

The important diagnostic was that learned relation phase error went nearly to zero:

```text
phase_mean_err=0.000
phase_max_err=0.001
```

Interpretation: when the task really is phase-structured, a tiny phase model can recover the rule, but enough data is needed for the phase effect to emerge.

## TinyStories Pair Probe

The initial natural-text pair probe was less favorable to quantum-native normalized states.

Frozen or normalized phase-state models stayed near chance or only weakly improved:

```text
frozen_phase       ~0.51
token_complex      ~0.55 at 8M
real_diag          ~0.73 at 8M
```

An unconstrained complex bilinear diagnostic baseline was competitive:

```text
8M examples:
complex_diag          0.7380 params=11383812
real_diag_wide        0.7403 params=11383810
complex_diag_halfdim  0.7306 params=5691908
real_diag             0.7284 params=5691906
```

Interpretation: complex bilinear geometry is competitive with real bilinear geometry, but the normalized Born-overlap tokenizer/state-prep family is too constrained for this noisy co-occurrence task.

## TinyStories Attention Probe

The attention probe produced the strongest natural-text signal so far.

At smaller data sizes, real attention won:

```text
250k:
real_attention_wide       0.6978
complex_attention         0.6267
complex_attention_halfdim 0.6112
```

At 4M examples, complex attention overtook both real baselines:

```text
4M:
complex_attention         0.8761 params=5693444
complex_attention_halfdim 0.8665 params=2846724
real_attention_wide       0.8387 params=5957122
real_attention            0.8245 params=2913026
```

Ablations showed that trainable phase rotations matter:

```text
4M:
complex_attention         0.8761
complex_attention_born    0.8512
complex_attention_nophase 0.6993
```

Interpretation: complex numbers appear more valuable in the interaction/attention component than in the tokenizer. The phase effect is scale dependent and depends strongly on learned phase rotations. Signed complex readout works better than Born-style squared readout for this next-token discrimination task, although Born-style readout remains useful at scale.

### Depth Behavior

Stacked attention shows a non-monotonic depth regime.

The regular complex stack peaks around 8 layers and then degrades:

```text
4M examples:
complex_stack L8   0.8731
complex_stack L12  0.8407
complex_stack L16  0.6993
complex_stack L24  0.7062
complex_stack L32  0.7697
```

The phase-floor variant initially hurts, but unexpectedly enters a better high-depth regime:

```text
complex_floor L8   0.8597
complex_floor L12  0.8160
complex_floor L16  0.6977
complex_floor L24  0.8147
complex_floor L32  0.8819
```

At 32 layers, `complex_attention_stacked_floor` beats the real stacked baseline:

```text
complex_floor L32  0.8819
real_stack L32     0.8697
```

The per-layer traces suggest this is not simply "larger phase is better." The floor model at 32 layers has lower aggregate phase magnitude than the regular stack, but has a late-layer phase ramp and lower residual mix. This points toward a depth-dependent phase transport regime that needs more focused sweeps.

The scheduled late-phase variant initially underperforms the floor variant, but crosses over at high depth:

```text
4M examples:
scheduled L24  0.7177
scheduled L28  0.7419
scheduled L32  0.8019
scheduled L36  0.8790
scheduled L40  0.8847
```

At 40 layers, scheduled complex attention beats both floor complex attention and real stacked attention:

```text
scheduled_complex L40  0.8847
floor_complex L40      0.8787
real_stack L40         0.8714
```

This supports the "transport then transform" hypothesis: many early layers stay low-mix and low-phase, while later layers form a phase ramp.

## Current Research Direction

The evidence now points away from forcing the tokenizer to be quantum-native early, and toward complex-valued contextual interaction layers:

- use learnable lexical embeddings,
- introduce complex Q/K/V-style interaction,
- preserve and ablate phase rotations,
- compare against real models at matched parameter and data budgets,
- treat Born-style readout as one measurement choice, not the default for every task.

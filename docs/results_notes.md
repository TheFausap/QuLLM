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

The high-depth sweep showed this was not a single-depth accident. Complex models stabilize above the real baseline:

```text
4M examples:
floor L44      0.8893
floor L48      0.8865
floor L56      0.8891
floor L64      0.8888
scheduled L44  0.8840
scheduled L48  0.8864
scheduled L56  0.8879
scheduled L64  0.8867
real L44       0.8702
real L48       0.8709
real L56       0.8709
real L64       0.8695
```

The peak-depth control run separated learned phase transport from several
weaker alternatives. The active-phase floor model stayed near 0.888 accuracy,
while zero-phase and frozen-phase variants all collapsed to roughly 0.66-0.68:

```text
4M examples, mean over seeds 31/37/43:
L44 floor          0.887956
L44 scheduled      0.885909
L44 real           0.868973
L44 decohere       0.676707
L44 free_mix       0.663848
L44 frozen_phase   0.680288

L56 floor          0.887972
L56 scheduled      0.887680
L56 real           0.868396
L56 decohere       0.676215
L56 free_mix       0.669793
L56 frozen_phase   0.675879

L64 floor          0.888943
L64 scheduled      0.887731
L64 real           0.868027
L64 decohere       0.675752
L64 free_mix       0.671977
L64 frozen_phase   0.677192
```

Paired gaps were stable:

```text
floor - real stacked:        +0.019 to +0.021
scheduled - real stacked:    +0.017 to +0.020
floor - decohere:            +0.211 to +0.213
floor - free_mix:            +0.217 to +0.224
floor - frozen_phase:        +0.208 to +0.212
```

Interpretation: learned phase rotations are load-bearing inside this deep
complex stack. Removing phase, removing the mix floor, or keeping only fixed
random phase transport all fail. This does not mean the general phase-free
ceiling is 0.676, because `real_attention_stacked` already reaches about 0.868.
The conservative cross-architecture advantage on this task is therefore the gap
to the best competent phase-free control, currently about +0.02, while the
within-complex-stack value of trainable phase transport is about +0.21. For
floor and scheduled variants, treat `mix_mean` as the historical raw gate trace
and `mix_effective_mean` as the actual residual mix used in the forward pass.
For decohered rows, `phase_active=0` means phase rotations are skipped by the
forward pass.

## Current Research Direction

The evidence now points away from forcing the tokenizer to be quantum-native early, and toward complex-valued contextual interaction layers:

- use learnable lexical embeddings,
- introduce complex Q/K/V-style interaction,
- preserve and ablate phase rotations,
- compare against real models at matched parameter and data budgets,
- treat Born-style readout as one measurement choice, not the default for every task.

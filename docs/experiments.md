# Experiments

For a running interpretation of observed results, see [results_notes.md](results_notes.md).

## Phase Relation Scaling

`experiments/phase_relation_scaling.py` tests whether a phase/unitary inductive bias becomes more valuable as data grows.

The teacher generates labels from:

```text
score(left, relation, right) = |<left| U_relation |right>|^2
```

where `U_relation` is a diagonal unitary. The students are:

- `phase_unitary`: normalized complex token states plus learned diagonal relation phases.
- `real_diag`: real-valued diagonal bilinear baseline.
- `amplitude_only`: normalized nonnegative amplitudes with no complex phase.
- `phase_scrambled`: complex token phases, but relation phases disabled.

The important comparison is not only final accuracy. Watch the scaling curve as training size increases. A useful phase effect should appear as a widening gap between `phase_unitary` and the ablations.

## MacBook Run

```bash
python3 -m pip install -r requirements.txt
python3 experiments/phase_relation_scaling.py --config configs/macbook_phase_sweep.json
```

On Apple Silicon, the script uses `mps` automatically when available. You can force CPU with:

```bash
python3 experiments/phase_relation_scaling.py --config configs/macbook_phase_sweep.json --device cpu
```

## DGX Run

```bash
python3 -m pip install -r requirements.txt
python3 experiments/phase_relation_scaling.py --config configs/dgx_phase_sweep.json --device cuda
```

The DGX config is deliberately larger:

- 65,536 token states,
- 128 complex dimensions,
- 32 relation operators,
- up to 1,048,576 training triples.

## Reading Results

The script writes CSV results to:

```text
runs/phase_relation_scaling.csv
```

Summarize the run with:

```bash
python3 experiments/summarize_phase_results.py runs/phase_relation_scaling.csv
```

The expected sanity pattern is:

1. `phase_unitary` should learn the teacher fastest.
2. `phase_scrambled` should underperform because relation phase is disabled.
3. `amplitude_only` should struggle because the teacher's labels depend on interference.
4. `real_diag` may improve with scale but should need more data or parameters.

If `phase_unitary` does not separate from the ablations on this synthetic task, the current encoding hypothesis is weak and should be revised before natural-language tests.

## Structured Phase Scaling

`experiments/structured_phase_scaling.py` is the better first probe for the tokenizer idea.

It creates synthetic tokens with three factors:

- `group`: amplitude support, analogous to a semantic or morphological basis.
- `phase_class`: a discrete relative phase, analogous to relational role.
- `variant`: surface identity; test examples use held-out variants.

The teacher marks a pair positive when two tokens share a group and their phase classes match the relation's phase shift. This gives us an OOV-like test: a model must use the structured token factors rather than memorize token IDs.

Run on Mac:

```bash
python3 experiments/structured_phase_scaling.py --config configs/macbook_structured_phase_sweep.json
python3 experiments/summarize_phase_results.py runs/structured_phase_scaling.csv
```

Run on DGX:

```bash
python3 experiments/structured_phase_scaling.py --config configs/dgx_structured_phase_sweep.json --device cuda
python3 experiments/summarize_phase_results.py runs/structured_phase_scaling.csv
```

Expected pattern:

- `phase_feature` should learn quickly because the correct hypothesis is a tiny set of relation phase shifts.
- `phase_margin` uses the same phase hypothesis with a sharper discrete-class measurement threshold.
- `phase_margin_fixed` is the cleanest pure phase-rule model: relation phases are trainable, while the measurement scale and threshold are fixed.
- `no_relation_phase` should fail on relation-dependent cases.
- `amplitude_feature` should detect same-group pairs but fail on wrong-phase same-group negatives.
- `real_feature_mlp` is a strong classical feature baseline. If it ties the phase model, the signal is not uniquely quantum-like; compare data efficiency and parameter count.

The CSV includes `trainable_params` and, for phase-relation models, `phase_mean_error` / `phase_max_error` in radians against the synthetic teacher's relation phase shifts. Use those diagnostics to distinguish optimization failure from representational limits.

For phase-margin models, the CSV also includes `phase_rule_accuracy`, which applies the learned relation phases with a fixed deterministic threshold and ignores learned scale/bias calibration. If `phase_rule_accuracy` is higher than normal accuracy, the phase representation is correct and the readout calibration is the remaining failure mode. In current runs, `phase_margin_fixed` is the preferred reference because it tests the phase rule without extra calibration degrees of freedom.

## TinyStories Pair Probe

`experiments/tinystories_pair_probe.py` is the first less-controlled natural-text experiment.

It streams `roneneldan/TinyStories` from Hugging Face and converts stories into binary local relation triples:

```text
(word_i, signed_relative_position, word_j)
```

Positive examples are real local pairs from the text. Negative examples replace `word_j` with a unigram-sampled random token. This is a noisy word-relation task, closer to language modeling than the synthetic phase benchmark while still cheap enough for scale sweeps.

Run on Mac:

```bash
python3 experiments/tinystories_pair_probe.py --config configs/macbook_tinystories_pair_probe.json
python3 experiments/summarize_phase_results.py runs/tinystories_pair_probe.csv
```

The configs cache generated vocabulary and train/test tensors under `runs/cache/`. The first run still streams and builds examples; later runs reuse the cached `.pt` file as long as dataset-building metadata matches.

Run on DGX:

```bash
python3 experiments/tinystories_pair_probe.py --config configs/dgx_tinystories_pair_probe.json --device cuda
python3 experiments/summarize_phase_results.py runs/tinystories_pair_probe.csv
```

This probe compares:

- `frozen_phase`: fixed complex feature states plus learned relation phases.
- `frozen_amplitude`: same fixed feature amplitudes with phase removed.
- `token_phase`: fixed feature-prepared amplitudes and base phases, plus learned per-token phase residuals and relation phases.
- `token_phase_lowrank`: lower-parameter phase residuals factorized through a small rank.
- `token_complex`: learned per-token amplitude and phase residuals, with normalized complex states.
- `token_complex_role`: separate left/right amplitude and phase residuals for asymmetric word-role behavior.
- `token_complex_signed`: normalized complex states with a signed complex inner-product readout instead of Born probability.
- `token_complex_role_signed`: role-specific signed complex readout.
- `complex_diag`: unconstrained complex bilinear diagnostic baseline. This is less quantum-native, but tests whether complex-valued bilinear geometry can match the real baseline before adding unitary constraints back.
- `complex_diag_halfdim`: roughly parameter-matched to `real_diag`.
- `real_diag`: learned real token embeddings and relation diagonals.
- `real_diag_wide`: roughly parameter-matched to `complex_diag`.

The important first question is whether frozen phase features beat frozen amplitude features. If they do not, compare `token_complex_signed` / `token_complex_role_signed` against `real_diag`: this asks whether a quantum-native normalized complex state parameterization can learn natural lexical structure once the readout preserves signed phase information.

## TinyStories Attention Probe

`experiments/tinystories_attention_probe.py` tests complex numbers inside an attention-like interaction rather than inside the tokenizer.

The task is binary next-token discrimination:

```text
(context window, candidate next token) -> real candidate or unigram negative
```

Models:

- `real_attention`: real candidate-to-context attention.
- `complex_attention`: complex candidate-to-context attention with phase rotations and signed complex readout.
- `complex_attention_halfdim`: roughly parameter-matched to `real_attention`.
- `complex_attention_nophase`: disables trainable phase rotations to test whether phase dynamics matter.
- `complex_attention_born`: uses Born-style squared compatibility/readout to compare against signed complex readout.
- `real_attention_stacked`: several candidate-conditioned real attention blocks.
- `complex_attention_stacked`: several candidate-conditioned complex phase-attention blocks.
- `real_attention_wide`: roughly parameter-matched to `complex_attention`.

Run on DGX:

```bash
python3 experiments/tinystories_attention_probe.py --config configs/dgx_tinystories_attention_probe.json --device cuda
python3 experiments/summarize_phase_results.py runs/tinystories_attention_probe.csv
```

Depth sweep:

```bash
python3 experiments/tinystories_attention_probe.py --config configs/dgx_tinystories_attention_depth_sweep.json --device cuda
python3 experiments/summarize_phase_results.py runs/tinystories_attention_depth_sweep.csv
```

Focused powers-of-two depth sweep at 4M examples:

```bash
python3 experiments/tinystories_attention_probe.py --config configs/dgx_tinystories_attention_depth_powers.json --device cuda
python3 experiments/summarize_phase_results.py runs/tinystories_attention_depth_powers.csv
```

Extended depth sweep around the 16-layer collapse:

```bash
python3 experiments/tinystories_attention_probe.py --config configs/dgx_tinystories_attention_depth_extended.json --device cuda
python3 experiments/summarize_phase_results.py runs/tinystories_attention_depth_extended.csv
```

Phase-floor variant sweep:

```bash
python3 experiments/tinystories_attention_probe.py --config configs/dgx_tinystories_attention_floor_sweep.json --device cuda
python3 experiments/summarize_phase_results.py runs/tinystories_attention_floor_sweep.csv
```

High-depth phase-floor sweep:

```bash
python3 experiments/tinystories_attention_probe.py --config configs/dgx_tinystories_attention_floor_highdepth.json --device cuda
python3 experiments/summarize_phase_results.py runs/tinystories_attention_floor_highdepth.csv
```

Scheduled late-phase sweep:

```bash
python3 experiments/tinystories_attention_probe.py --config configs/dgx_tinystories_attention_scheduled_sweep.json --device cuda
python3 experiments/summarize_phase_results.py runs/tinystories_attention_scheduled_sweep.csv
```

Scheduled high-depth sweep:

```bash
python3 experiments/tinystories_attention_probe.py --config configs/dgx_tinystories_attention_scheduled_highdepth.json --device cuda
python3 experiments/summarize_phase_results.py runs/tinystories_attention_scheduled_highdepth.csv
```

Decohere and multi-seed control:

```bash
python3 experiments/tinystories_attention_probe.py --config configs/dgx_tinystories_attention_decohere_seeds.json --device cuda
python3 experiments/summarize_phase_results.py runs/tinystories_attention_decohere_seeds.csv
```

Best-depth decohere and multi-seed control:

```bash
python3 experiments/tinystories_attention_probe.py --config configs/dgx_tinystories_attention_decohere_bestdepths.json --device cuda
python3 experiments/summarize_phase_results.py runs/tinystories_attention_decohere_bestdepths.csv
```

The complex stack diagnostics report both the historical raw gate (`mix_mean`,
`mix_by_layer`) and the actual residual mix used by the block
(`mix_effective_mean`, `mix_effective_by_layer`). Use the effective mix columns
when comparing floor and scheduled variants. `phase_active=0` marks the
decohered twin where phase parameters remain in the model shape but are not used
by the forward pass.

# Experiments

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
- `real_diag`: learned real token embeddings and relation diagonals.

The important first question is whether frozen phase features beat frozen amplitude features. If they do not, compare `token_phase` / `token_phase_lowrank` against `real_diag`: this asks whether a quantum-native state parameterization can learn natural lexical structure with fewer parameters than unconstrained real token embeddings.

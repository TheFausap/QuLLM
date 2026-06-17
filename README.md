# QuLLM

QuLLM is a research sketch for language models whose internal computation is built around complex-valued states and unitary transformations, rather than real lookup embeddings followed by unconstrained affine layers.

The first target is modest: define a quantum-native input path that can be tested on toy linguistic relations before trying to scale it into a transformer replacement.

## Current Hypothesis

Classical tokenization and embedding make language enter the model as arbitrary discrete IDs. A quantum-native path should instead make the input a normalized complex state from the beginning:

- amplitudes encode feature support across a finite Hilbert basis,
- phases encode token identity, position, and relational roles,
- contextual interactions are represented by unitary phase shifts or norm-preserving mixers,
- compatibility is read out with Born-rule overlap, `|<a|b>|^2`.

This does not claim a quantum hardware advantage yet. It is a classical simulation of a quantum-inspired architecture, useful for identifying what the encoding should preserve before hardware constraints dominate the design.

## Files

- [docs/research_agenda.md](docs/research_agenda.md) captures the architecture directions, nearby literature, and experimental plan.
- [docs/experiments.md](docs/experiments.md) describes the first dataset-size scaling experiment.
- [src/qnative_encoding.py](src/qnative_encoding.py) implements a small dependency-free complex state toolkit.
- [examples/toy_phase_relations.py](examples/toy_phase_relations.py) demonstrates relative-phase relation scoring.
- [experiments/phase_relation_scaling.py](experiments/phase_relation_scaling.py) trains phase/unitary and baseline models across dataset sizes.
- [tests/test_qnative_encoding.py](tests/test_qnative_encoding.py) checks norm preservation and the toy relational behavior.

## Run

```bash
python3 examples/toy_phase_relations.py
python3 -m unittest discover -s tests
```

For the PyTorch scaling experiment:

```bash
python3 -m pip install -r requirements.txt
python3 experiments/phase_relation_scaling.py --config configs/macbook_phase_sweep.json
python3 experiments/summarize_phase_results.py runs/phase_relation_scaling.csv
```

The more tokenizer-focused probe is:

```bash
python3 experiments/structured_phase_scaling.py --config configs/macbook_structured_phase_sweep.json
python3 experiments/summarize_phase_results.py runs/structured_phase_scaling.csv
```

The first less-controlled natural-text probe uses TinyStories:

```bash
python3 experiments/tinystories_pair_probe.py --config configs/macbook_tinystories_pair_probe.json
python3 experiments/summarize_phase_results.py runs/tinystories_pair_probe.csv
```

## Next Experiments

1. Compare feature-state encoding against ID lookup embeddings on a tiny synthetic grammar task.
2. Replace hard-coded relation phase operators with learned diagonal unitaries.
3. Add a unitary sequence block with position phases, relation phases, and Givens-style token mixing.
4. Decide whether the first serious implementation should be PyTorch complex tensors or JAX.
